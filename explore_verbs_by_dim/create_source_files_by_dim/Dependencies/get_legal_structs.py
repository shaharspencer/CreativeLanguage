import copy
import os.path

import spacy



import csv
from spacy.tokens import DocBin
import docopt

usage = '''
get_legal_structs CLI.
Usage:
    get_legal_structs.py
    get_legal_structs.py spacy_path csv_path as_list
'''

# TODO put correct file directories and names
# and output files

# make sure changing the doc arrangment didnt change anything else


class CreateWhCsv():
    # we want to find subsets of these for dependency list
    # dimension
    DEPS_FOR_LIST = {"nsubj", "nsubjpass", "ccomp", "csubj"
        , "xcomp", "prt", "dobj", "prep", "dative",
                          }
    # we want to find subsests of these for dependency set
    DEPS_FOR_SET = {"ccomp", "xcomp",
                         "prt", "dobj", "prep", "dative"}
    def __init__(self,  csv_path,
                 counter_csv_path,
                 model = "en_core_web_lg",
                 spacy_dir_path= r"C:\Users\User\OneDrive\Documents\CreativeLanguageOutputFiles\training_data\spacy_data\withought_context_lg_model",
                 spacy_file_path=r"data_from_first_50000_lg_model.spacy",
                 arrange_deps_as_list=True):
        self.nlp = spacy.load(model)

        self.spacy_path = os.path.join(spacy_dir_path, spacy_file_path)
        self.counter_csv_path = counter_csv_path
        self.arrange_deps_as_list = arrange_deps_as_list
        self.csv_path = csv_path

        self.possible_combs = set()
        self.count_structs = {}

        self.WH_WORDS = {"what", "which", "whatever", "who", "whom"
            , "whateva", "whoever", "whichever", "when", "whenever", "how"}
        # TODO are there more???
        self.illegal_heads = {"relcl", "advcl", "acl"}
        self.counter = 0
        self.dict_for_csv = {}
        self.doc_bin = DocBin().from_disk(self.spacy_path)

    def create_csv(self):
        self.create_csv_dict()
        self.write_dict_to_csv()
        if not self.arrange_deps_as_list:
            self.write_counter_colwise_csv_for_set()
        else:
            self.write_counter_dep_struct_csv()

    def verify_token_type(self, token)-> bool:
        if token.pos_ != "VERB" or not token.text.isalpha():
            return False
            # get dependencies
        return True

    def create_verb_set(self):
        self.verbs = set()
        # don't need to convert to list because it is an iterator
        for doc in self.doc_bin.get_docs(self.nlp.vocab):
            for token in doc:
                if token.pos_ != "VERB" or not token.text.isalpha():
                    continue
                self.verbs.add(token.lemma_.lower())

    # returns true if there is such descendanct
    def check_if_has_desc_which_is_wh(self, token: spacy.tokens)->bool:
        filtered_subtree = set(
            filter(lambda k: k.dep_ in DEP_LIST and k != token,
                   [t for t in token.subtree]))
        return \
            (len(set([t.text for t in filtered_subtree]
                     ).intersection(self.WH_WORDS)) != 0)

    # TODO check complex wh words
    """
    
    """
    def check_legal_token_deps(self, token: spacy.tokens,
                               token_dep_list: list[spacy.tokens])->bool:
        # check if it is a relcl
        def check_if_relcl()->bool:
            for child in [child for child in token.head.children]:
                if child == token and child.dep_ in self.illegal_heads:
                    return False
            return True

        # check if it has a core dependent which is a wh word and preceds the
        # subject
        def check_if_core_dep_wh_word()->bool:
            for dep_word in token_dep_list:
                if dep_word.dep_ in self.DEP_LIST\
                        and dep_word.i < token.i and \
                        dep_word.text in self.WH_WORDS:
                    return False
            return True
        #The verb has a core dependent which precedes the subject and the dependent
        # has a descendent (not necessarily immediate) which precedes it and is a wh-word
        def check_complex_dobj_wh_word()->bool:
            for dep_word in token_dep_list:
                if dep_word.dep_ in self.DEP_LIST \
                        and dep_word.i < token.i:
                    return not self.check_if_has_desc_which_is_wh(dep_word)
            return True
        # check that these are the types of children we want
        def check_token_dep_types():
            token_dep_types = [t.dep_ for t in token_dep_list]
            return set(token_dep_types).issubset(self.DEP_LIST)
        if  self.arrange_deps_as_list:
            return (check_if_relcl() and check_if_core_dep_wh_word() and
                    check_complex_dobj_wh_word() and check_token_dep_types())
        else:
            return (check_if_relcl() and check_token_dep_types())
    """
    checks:
    that token is of the type we want to add
    that token dependency list is of the type we want
    if either if false, we don't add token to dictionary
    @:param token - token to check whether to add
    @:param token_dep_list - token dependency list, after cleaning it
    via clean_token_children
    """
    def check_if_add_token(self, token: spacy.tokens, token_dep_list: list):
        if not self.verify_token_type(token):
            return False

        if not (self.check_legal_token_deps(token, token_dep_list)):
            return False
        return True
    """
    arranges deps as either a list or a set, depending on arrange_as_list
    arrange_as_list = True if we want to create dependency structure
    arrange_as_list = False if we want to create dependency set
    """
    def arrange_deps(self, token: spacy.tokens,
                     token_children: list[spacy.tokens],
                     arrange_as_list = True)->str:
        if (arrange_as_list):
            TokenDependencies = sorted(token_children + [token], key=lambda k:
            k.i)
            TokenDependencies = [x.dep_ if x != token else "V" for
                                 x in TokenDependencies]

            return "_".join(TokenDependencies)
        # currently sorted by default order
        else:
            return "_".join(set(sorted([x.dep_ for x in token_children])))
    """
    adds a single token to dictionary
    first checks the token is the type we want to add
    (is a verb, checks the deps are the type we want)
    then verifys token has a key in the dictionary
    then adds to dictionary
    """
    def add_token_to_dict(self, doc_index: int, sent_indx: int,
                          token: spacy.tokens)->bool:

        token_children = self.clean_token_children(token)
        # verify that we want to add this token
        if not self.check_if_add_token(token, token_children):
            return False
        # create key for token if not yet in dict keys
        self.add_verb_template_to_dict(token)
        # arrange dependencies
        token_dep_comb = self.arrange_deps(token, token_children, arrange_as_list=
        self.arrange_deps_as_list)

        self.possible_combs.add(token_dep_comb)
        dict_tuple = (token.text, doc_index, sent_indx,
                 token.sent.text.strip())

        if token_dep_comb in self.dict_for_csv[token.lemma_.lower()]:
            self.dict_for_csv[token.lemma_.lower()] \
                [token_dep_comb]["counter"] += 1
            self.dict_for_csv[token.lemma_.lower()] \
                [token_dep_comb]["instances"].add(dict_tuple)

        else:
            self.dict_for_csv[token.lemma_.lower()] \
                [token_dep_comb] = {"instances": set(), "counter": 1}
            self.dict_for_csv[token.lemma_.lower()] \
                [token_dep_comb]["instances"].add(dict_tuple)


    """
    adds a single doc object to dictionary
    each doc should only have one sentence
    """

    def add_doc_to_dict(self, doc):
        for sent in doc.sents:
            for token in sent:
               self.add_token_to_dict(doc.user_data["DOC_INDEX"],
                                      doc.user_data["SENT_INDEX"], token)


    def create_csv_dict(self):
        self.create_verb_set()
        # for verb in self.verbs:
        #     self.dict_for_csv[verb] = {}
        for doc in self.doc_bin.get_docs(self.nlp.vocab):
            self.add_doc_to_dict(doc)


    def write_dict_to_csv(self):
        with open(self.csv_path, 'w', encoding='utf-8', newline='') as f:
            # create struct for csv
            fieldnames = ["Lemma (V)", 'Verb form', "Dep struct",
                          "Sentence",
                          "Doc index", 'Sent index']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            d = {'Lemma (V)': 'Lemma (V)', 'Verb form': 'Verb form',
                 'Dep struct': 'Dep struct',
                 'Sentence': 'Sentence',
                 'Doc index': 'Doc index',
                 'Sent index': 'Sent index'}

            writer.writerow(d)

            self.write_all_rows(writer)



    def write_all_rows(self, writer):
        for word in self.dict_for_csv.keys():
            for comb in self.possible_combs:
                try:
                    for entry in self.dict_for_csv[word][comb]["instances"]:
                        lemma = word
                        verb_form = entry[0]
                        doc_index = entry[1]
                        sent_index = entry[2]
                        sentence = entry[3]
                        if comb == "":
                            comb = "NO_DEPS"
                        n_dict = {'Lemma (V)': lemma, 'Verb form': verb_form,
                                  'Sentence': sentence,
                                  'Dep struct': comb,
                                  'Doc index': doc_index,
                                  'Sent index': sent_index}
                        writer.writerow(n_dict)
                except KeyError:
                    pass

    """
    removes dependencies we don't want in dep list
    @:param clean_punct: remove punctuation dependency from childern
    """
    def clean_token_children(self, token, clean_punct = True) -> list:
        token_children = [child for child in token.children]
        if clean_punct:
            token_children = list(filter(lambda x: x.dep_ != "punct",
                                         token_children))
        return token_children

    """
    this function is used when we are analyzing the dependency set
    we are writing a csv with the percentages and counts of the 
    different dependency set across all verbs
    """
    def write_counter_colwise_csv_for_set(self):
        with open(self.counter_csv_path, 'w', encoding='utf-8', newline='') as f:
            clean_possible_combs = copy.deepcopy(self.possible_combs)
            clean_possible_combs.remove("")
            clean_possible_combs.add("NO_DEPS")
            fieldnames_for_combs = print_fieldnames(clean_possible_combs)
            fieldnames = ["Lemma (V)"] + list(fieldnames_for_combs.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            d = {"Lemma (V)": "Lemma (V)"}
            d.update(fieldnames_for_combs)
            writer.writerow(d)
            for word in self.dict_for_csv.keys():
                n_dict = {'Lemma (V)': word}
                sum_word = sum(
                    len(
                        self.dict_for_csv[word][comb]["instances"]
                    )
                    for comb in
                    self.dict_for_csv[word])
                for comb in self.possible_combs:
                    comb_name = comb
                    if comb == "":
                        comb_name = "NO_DEPS"
                    try:
                        counter = len(self.dict_for_csv[word][comb]["instances"])
                        n_dict[comb_name+"_COUNT"] = counter
                        n_dict[comb_name + "%"] = counter / sum_word

                    except KeyError:
                        n_dict[comb_name + "_COUNT"] = 0
                        n_dict[comb_name + "%"] = 0
                writer.writerow(n_dict)

    """
     this function is used when we are analyzing the dependency list
     we are writing a csv with the percentages and counts of the 
     different dependency lists across all verbs
     """
    def write_counter_dep_struct_csv(self):
        with open(self.counter_csv_path, 'w', encoding='utf-8',
                  newline='') as f:
            clean_possible_combs = copy.deepcopy(self.possible_combs)
            fieldnames_for_combs = print_fieldnames(clean_possible_combs)
            fieldnames = ["Lemma (V)"] + list(fieldnames_for_combs.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            d = {"Lemma (V)": "Lemma (V)"}
            d.update(fieldnames_for_combs)
            writer.writerow(d)
            for word in self.dict_for_csv.keys():
                n_dict = {'Lemma (V)': word}
                sum_word = sum(
                    len(
                        self.dict_for_csv[word][comb]["instances"]
                    )
                        for comb in
                    self.dict_for_csv[word])
                for comb in self.possible_combs:
                    try:
                        n_dict[comb + "_COUNT"] = \
                        len(self.dict_for_csv[word][comb]["instances"])
                        n_dict[comb + "%"] = \
                        len(self.dict_for_csv[word][comb]["instances"]) / sum_word

                    except KeyError:
                        n_dict[comb + "_COUNT"] = 0
                        n_dict[comb + "%"] = 0
                writer.writerow(n_dict)

    """
    each token needs a key in the final dictionary
    if the token.lemma_lower() is not yet a key, add as key
    """
    def add_verb_template_to_dict(self, token:spacy.tokens):
        if not token.lemma_.lower() in self.dict_for_csv.keys():
            self.dict_for_csv[token.lemma_.lower()] = {}


def print_fieldnames(given_lst: iter):
    dic = {}
    for fieldname in given_lst:
        dic[fieldname+"_COUNT"] = fieldname+"_COUNT"
        dic[fieldname+"%"] = fieldname + "%"
    return dic



if __name__ == '__main__':
    from datetime import datetime

    datetime = datetime.today().strftime('%Y_%m_%d')

    import datetime

    output_dir = r"C:\Users\User\OneDrive\Documents\CreativeLanguageOutputFiles\dependency_set_dimension"

    csv_path = "{n}_dependency_set_from_first_25000_posts_lg_sents.csv".format(
       n= datetime)

    counter_path = "{n}_dependency_set_from_first_25000_posts_lg_counter.csv".format(
       n=datetime)
    args = docopt.docopt(usage)
    if args["spacy_path"] and args["csv_path"]:
        createWhCsv = CreateWhCsv(args["spacy_path"], args["csv_path"],
                                  arrange_deps_as_list=args["as_list"])

    else:
        createWhCsv = CreateWhCsv(csv_path=os.path.join(output_dir, csv_path),
                                  counter_csv_path=os.path.join(output_dir, counter_path),
                                  )
    createWhCsv.create_csv()

