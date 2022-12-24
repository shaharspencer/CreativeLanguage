import copy

import spacy


nlp = spacy.load("en_core_web_trf")
import csv
from spacy.tokens import DocBin
import docopt

usage = '''
get_legal_structs CLI.
Usage:
    get_legal_structs.py
    get_legal_structs.py spacy_path csv_path as_list
'''


class CreateWhCsv():
    def __init__(self,
                 spacy_path=
                 r"C:\Users\User\PycharmProjects\CreativeLanguage\training_data\spacy_data\data_from_first_15000_posts.spacy",
                 csv_path="first_15000_posts_sents_arg_struct_dim.csv",
                 counter_csv_path = "counter_first_15000_posts_arg_struct_dim.csv",
                 arrange_deps_as_list = False):
        self.counter_csv_path = counter_csv_path
        self.arrange_deps_as_list = arrange_deps_as_list
        self.csv_path = csv_path
        self.spacy_path = spacy_path
        self.possible_combs = set()
        self.count_structs = {}
        self.DEP_LIST = {"nsubj", "nsubjpass", "ccomp", "csubj"
            , "xcomp", "prt", "dobj", "prep", "dative",
                         }
        self.WH_WORDS = {"what", "which", "whatever", "who", "whom"
            , "whateva", "whoever", "whichever", "when", "whenever", "how"}
        # TODO are there more???
        self.illegal_heads = {"relcl", "advcl", "acl"}

        self.dict_for_csv = {}
        self.doc_bin = DocBin().from_disk(self.spacy_path)

    def create_csv(self):
        self.create_csv_dict()
        self.write_dict_to_csv()
        # self.write_counter_csv()
        if not self.arrange_deps_as_list:
            self.write_counter_colwise_csv_for_set()
        # else:
        #     self.write_counter_dep_struct_csv()

    def verify_token_type(self, token)-> bool:
        if token.pos_ != "VERB" or not token.text.isalpha():
            return False
            # get dependencies
        return True

    def create_verb_set(self):
        self.verbs = set()
        for doc in list(self.doc_bin.get_docs(nlp.vocab)):
            for token in doc:
                if token.pos_ != "VERB" or not token.text.isalpha():
                    continue
                self.verbs.add(token.lemma_.lower())

    # returns true if there is such descendanct
    def check_if_has_desc_which_is_wh(self, token: spacy.tokens)->bool:
        filtered_subtree = set(
            filter(lambda k: k.dep_ in self.DEP_LIST and k != token,
                   [t for t in token.subtree]))
        return \
            (len(set([t.text for t in filtered_subtree]
                     ).intersection(self.WH_WORDS)) != 0)

    # TODO check complex wh words
    def check_legal_token_deps(self, token: spacy.tokens,
                               token_dep_list: list[spacy.tokens]):
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

        return (check_if_relcl() and check_if_core_dep_wh_word() and
                check_complex_dobj_wh_word() and check_token_dep_types())

    def check_if_add_token(self, token: spacy.tokens, token_dep_list: list):
        if not self.verify_token_type(token):
            return False

        if not (self.check_legal_token_deps(token, token_dep_list)):
            return False
        return True

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
            return "_".join(list(sorted([x.dep_ for x in token_children])))

    def add_token_to_dict(self, doc_index: int, sent_indx: int,
                          token: spacy.tokens)->bool:
        token_children = self.clean_token_children(token)
        # verify that we want to add this token
        if not self.check_if_add_token(token, token_children):
            return False
        # arrange dependencies
        token_dep_comb = self.arrange_deps(token, token_children, arrange_as_list=
        self.arrange_deps_as_list)
        self.possible_combs.add(token_dep_comb)
        dict_tuple = (token.text, doc_index, sent_indx,
                 token.sent.text)

        if not token_dep_comb in self.dict_for_csv[token.lemma_.lower()]:
            self.dict_for_csv[token.lemma_.lower()] \
                [token_dep_comb] = {"instances": set(), "counter": 0}

        self.dict_for_csv[token.lemma_.lower()] \
            [token_dep_comb]["instances"].add(dict_tuple)
        self.dict_for_csv[token.lemma_.lower()] \
            [token_dep_comb]["counter"] += 1

    def add_doc_to_dict(self, doc: nlp, doc_index):
        for j, sent in enumerate(doc.sents):
            for token in sent:
               self.add_token_to_dict(doc_index, j, token)

    def create_csv_dict(self):
        self.create_verb_set()
        for verb in self.verbs:
            self.dict_for_csv[verb] = {}
        doc_counter = 0
        for doc in list(self.doc_bin.get_docs(nlp.vocab)):
            self.add_doc_to_dict(doc, doc_counter)
            doc_counter += 1

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
        counter = 0
        for word in self.dict_for_csv.keys():
            for comb in self.possible_combs:
                try:
                    for entry in self.dict_for_csv[word][comb]["instances"]:
                        counter += 1
                        #(token.text, doc_index, sent_indx,
                 #token.sent.text)
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

        print(counter)

    def clean_token_children(self, token, clean_punct = True) -> list:
        token_children = [child for child in token.children]
        if clean_punct:
            token_children = list(filter(lambda x: x.dep_ != "punct", token_children))
        return token_children

    # def write_counter_csv(self):
    #     with open(self.counter_csv_path, 'w', encoding='utf-8', newline='') as f:
    #         fieldnames = ["Lemma (V)", "Dep struct",
    #                       "Dep struct count"]
    #         writer = csv.DictWriter(f, fieldnames=fieldnames)
    #         d = {'Lemma (V)': 'Lemma (V)',
    #              'Dep struct': 'Dep struct',
    #              "Dep struct count":
    #                  "Dep struct count"}
    #         writer.writerow(d)
    #         for word in self.dict_for_csv.keys():
    #             for comb in self.possible_combs:
    #                 try:
    #                     comb_count = self.dict_for_csv[word][comb]["counter"]
    #                     if comb == "":
    #                         comb = "NO_DEPS"
    #
    #                     n_dict = {'Lemma (V)': word,
    #
    #                       'Dep struct': comb,
    #                               "Dep struct count": comb_count
    #                      }
    #                     writer.writerow(n_dict)
    #                 except KeyError:
    #                     n_dict = {'Lemma (V)': word,
    #
    #                               'Dep struct': comb,
    #                               "Dep struct count": 0
    #                               }
    #                     writer.writerow(n_dict)

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
                sum_word = sum(self.dict_for_csv[word][comb]["counter"] for comb in
                               self.dict_for_csv[word])
                for comb in self.possible_combs:
                    comb_name = comb
                    if comb == "":
                        comb_name = "NO_DEPS"
                    try:
                        n_dict[comb_name+"_COUNT"] = self.dict_for_csv[word][comb]["counter"]
                        n_dict[comb_name + "%"] = self.dict_for_csv[word][comb]["counter"] / sum_word

                    except KeyError:
                        n_dict[comb_name + "_COUNT"] = 0
                        n_dict[comb_name + "%"] = 0
                writer.writerow(n_dict)

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
                    self.dict_for_csv[word][comb]["counter"] for comb in
                    self.dict_for_csv[word])
                for comb in self.possible_combs:
                    comb_name = comb
                    try:
                        n_dict[comb_name + "_COUNT"] = \
                        self.dict_for_csv[word][comb]["counter"]
                        n_dict[comb_name + "%"] = \
                        self.dict_for_csv[word][comb]["counter"] / sum_word

                    except KeyError:
                        n_dict[comb_name + "_COUNT"] = 0
                        n_dict[comb_name + "%"] = 0
                writer.writerow(n_dict)


def print_fieldnames(given_lst: iter):
    dic = {}
    for fieldname in given_lst:
        dic[fieldname+"_COUNT"] = fieldname+"_COUNT"
        dic[fieldname+"%"] = fieldname + "%"
    return dic

if __name__ == '__main__':
    args = docopt.docopt(usage)
    if args["spacy_path"] and args["csv_path"]:
        createWhCsv = CreateWhCsv(args["spacy_path"], args["csv_path"],
                                  arrange_deps_as_list=args["as_list"])
    else:
        createWhCsv = CreateWhCsv()
    createWhCsv.create_csv()







