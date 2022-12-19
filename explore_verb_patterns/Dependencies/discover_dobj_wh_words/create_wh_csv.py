import os
import spacy
import itertools
nlp = spacy.load("en_core_web_trf")
import csv
from spacy.tokens import DocBin
import docopt



usage = '''
Processor CLI.
Usage:
    create_wh_csv.py
    create_wh_csv.py <spacy_path> <csv_path>
'''


class CreateWhCsv():
    def __init__(self,
                 spacy_path=
                 r"C:\Users\User\PycharmProjects\CreativeLanguage\training_data\spacy_data\data_from_first_15000_posts.spacy",
                 csv_path = "wh_csv_first_15000_posts.csv"):
        self.csv_path = csv_path
        self.spacy_path = spacy_path
        self.possible_combs = set()
        self.DEP_LIST = {"dobj", "nsubj", "ccomp", "xcomp",
                            "prt", "prep", "pobj",
                            "dative", "punct"}
        self.dict_for_csv = {}
        self.doc_bin = DocBin().from_disk(self.spacy_path)


    def create_csv(self):
        self.create_csv_dict()
        self.write_dict_to_csv()

    def verify_token(self, token):
        if token.pos_ != "VERB" or not token.text.isalpha():
            return False
            # get dependencies

        TokenDepTypes = [child.dep_ for child in list(token.children)]
        if not TokenDepTypes:
            return False

        if not TokenDepTypes[0] == "dobj":
                return False
        TokenDeps = [child for child in list(token.children)]

        if TokenDeps[0].i > token.i:
            return False

        # also clean spaces at begin of sents
        # BUG!!!!!!!!! not combs with v at begin
        # if token.i < TokenDepTypes[0].i:
        #     return False
        if not set(TokenDepTypes).issubset(self.DEP_LIST):
            return False
        return TokenDepTypes

    def get_dep_comb_and_word(self, token):
        TokensWithDep = [child for child in list(token.children)]
        TokenDependencies = sorted(TokensWithDep + [token], key=lambda k:
        k.i)
        TokenDependencies = [x.dep_ if x != token else "V" for
                             x in TokenDependencies]
        CombName = "_".join(TokenDependencies).replace("punct_", ""). \
            replace("_punct", ""). \
            replace("punct_", "").replace("punct", "")
        self.possible_combs.add(CombName)
        return CombName, TokensWithDep[0]

    def create_verb_set(self):
        self.verbs = set()
        for doc in list(self.doc_bin.get_docs(nlp.vocab)):
            for token in doc:
                if token.pos_ != "VERB" or not token.text.isalpha():
                    continue
                self.verbs.add(token.lemma_.lower())

    def get_token_father(self, token):
        for token_to_check in token.sent:
            # get list of token children
            dep_list = token_to_check.children
            for dep in dep_list:
                if dep == token and dep.dep_ == "acl":
                    return "acl", token_to_check.text
                if dep == token and dep.dep_ == "relcl":
                    return "relcl", token_to_check.text
                if dep == token and dep.dep_ == "advcl":
                    return "advcl", token_to_check.text
        return None, None


    def add_doc_to_dict(self, doc, doc_index):
        for j, sent in enumerate(doc.sents):
            for token in sent:
                if not self.verify_token(token):
                    continue

                token_father_type, token_father_text = self.get_token_father(token)
                token_dep_comb, word_which_is_dobj = \
                    self.get_dep_comb_and_word(token)

                # if need to add key to dict
                if not token_dep_comb in self.dict_for_csv[token.lemma_.lower()]:
                    self.dict_for_csv[token.lemma_.lower()][token_dep_comb] = \
                        {"instances": []}
                # if does not have relcl or acl father
                if not token_father_type:
                    self.dict_for_csv[token.lemma_.lower()] \
                        [token_dep_comb]["instances"].append(
                        (token.text, doc_index, j,
                         sent.text, word_which_is_dobj, "none", "none"))

                # if does have
                else:
                    self.dict_for_csv[token.lemma_.lower()] \
                        [token_dep_comb]["instances"].append(
                        (token.text, doc_index, j,
                         sent.text, word_which_is_dobj,
                         token_father_type, token_father_text))


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
            fieldnames = ["Lemma (V)", 'Verb form', "Dep form (dobj)",
                          "Sentence", "Dep struct", 'Father', 'Father form',
                          "Doc index", 'Sent index']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            d = {'Lemma (V)': 'Lemma (V)', 'Verb form': 'Verb form',
             'Dep form (dobj)': 'Dep form (dobj)', 'Sentence': 'Sentence',
             'Dep struct': 'Dep struct', 'Father': 'Father', 'Father form':
                     'Father form',
                     'Doc index': 'Doc index',
                          'Sent index': 'Sent index', }

            writer.writerow(d)

            self.write_all_rows(writer)


    def write_all_rows(self, writer):
        for word in self.dict_for_csv.keys():
            for comb in self.possible_combs:
                try:
                    for entry in self.dict_for_csv[word][comb]["instances"]:

                        #(token.text, doc_index, j,
                        # sent.text, word_which_is_dobj,
                        # token_father_type, token_father_text)
                        lemma = word
                        verb_form = entry[0]
                        dep_form = entry[4]
                        doc_index = entry[1]
                        sent_index = entry[2]
                        sentence = entry[3]
                        father_type = entry[5]
                        father_name = entry[6]

                        n_dict = {'Lemma (V)': lemma, 'Verb form': verb_form,

                        'Dep form (dobj)': dep_form, 'Sentence': sentence,
                        'Dep struct': comb, 'Father': father_type,
                                  'Father form': father_name,
                                  'Doc index':doc_index,
                                  'Sent index': sent_index}
                        writer.writerow(n_dict)
                except KeyError:
                    pass



def print_fieldnames():
    dic = {}
    for fieldname in ["Lemma (V)", "Dep form (dobj)", "Sentence", "Dep struct"]:
        dic[fieldname] = fieldname
    print(dic)



if __name__ == '__main__':
    args = docopt.docopt(usage)
    if args["<spacy_path>"] and args["<csv_path>"]:
        createWhCsv = CreateWhCsv(args["<spacy_path>"], args["<csv_path>"])
    else:
        createWhCsv = CreateWhCsv()
    createWhCsv.create_csv()


