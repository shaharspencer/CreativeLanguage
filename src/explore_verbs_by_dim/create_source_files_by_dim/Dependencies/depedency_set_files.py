import spacy
from docopt import docopt

from spacy.tokens import DocBin

import csv

import os

from src.explore_verbs_by_dim.create_source_files_by_dim.Dependencies.abstract_dependency_files import \
    DependencyDimensionFiles

import utils.path_configurations as paths


usage = '''
dependency_set_files CLI.
Usage:
    dependency_set_files.py <spacy_file_name> <num_of_posts>
'''




class DependencySetFiles(DependencyDimensionFiles):
    def __init__(self,spacy_file_path,
                 model="en_core_web_lg",
                 spacy_dir_path=r"withought_context_lg_model",
                 ):

        # TODO make sure these are correct deps
        # TODO make sure code is proper with proper tests etc
        # TODO sanity checks with counter and sentences csv's
        self.DEP_LIST = {"ccomp", "xcomp",
                        "prt", "dobj", "prep", "dative"}

        self.ILLEGAL_HEADS = {"relcl", "advcl", "acl"}

        self.irrelevant_parts_of_speech = {"punct", "SPACE"}

        # initialize super
        DependencyDimensionFiles.__init__(self, model=model, spacy_dir_path=
        spacy_dir_path, spacy_file_path=
                                          spacy_file_path)

    """
    removes dependencies we don't want in dep list
    @:param clean_punct: remove punctuation dependency from childern
    """

    def clean_token_children(self, token, clean_punct=True) -> list:
        token_children = [child for child in token.children]
        if clean_punct:
            token_children = list(filter(lambda x: x.dep_ != "punct" and x.pos_ !=
            "SPACE",
                                         token_children))
        return token_children

    """
     this method checks if the token itself is of the type we want to use
     needs to be a verb and satisfy token.text.isalpha
    """
    def verify_token_type(self, token) -> bool:
        if token.pos_ != "VERB" or not token.text.isalpha():
            return False
        return True

    """
        this method checks whether the verb is a relcl or an acl.
        if so returns false because we do not want to use
        this type of verb
        @:param token: token to check
    """
    # TODO: if father is relcl or acl, do we want to disregard it?
    def check_if_relcl(self, token: spacy.tokens) -> bool:
        if token.head.pos_ in self.irrelevant_parts_of_speech:
            return True
        for child in [child for child in token.head.children]:
            if child == token and child.dep_ in self.ILLEGAL_HEADS:
                return False
        return True

    """
    checks that this verb has a cleaned dependency set that is a 
    subset of the dependencies we are looking for
    @:param token_dep_list: cleaned dependency set for token
    :return True if dependency list is subset
            False else
    """
    def check_token_dep_types(self, token_dep_list: list[spacy.tokens]):
        token_dep_types = [t.dep_ for t in token_dep_list]
        return set(token_dep_types).issubset(self.DEP_LIST)

    """
        arranges deps as a set
        @:param token: the verb
        @token_children: cleaned token dependencies
    """
    def arrange_deps(self, token: spacy.tokens,
                     token_children: list[spacy.tokens]) -> str:
        return "_".join(set(sorted([x.dep_ for x in token_children])))


    def check_legal_token_deps(self, token: spacy.tokens,
                               token_dep_list: list[spacy.tokens]) -> bool:
        return self.check_if_relcl(token) and self.check_token_dep_types(
            token_dep_list)

    """
     this function is used when we are analyzing the dependency list
     we are writing a csv with the percentages and counts of the 
     different dependency sets across all verbs
     """

    def write_counter_csv(self, counter_csv_name):

        output_path = os.path.join(paths.files_directory,
                                   paths.dependency_set_directory,
                                   paths.dependency_set_source_files,
                                   counter_csv_name)
        import copy
        with open(output_path, 'w', encoding='utf-8',
                  newline='') as f:
            clean_possible_combs = copy.deepcopy(self.possible_combs)
            clean_possible_combs.remove("")
            clean_possible_combs.add("NO_DEPS")
            fieldnames_for_combs = self.print_fieldnames(clean_possible_combs)
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
                        counter = len(
                            self.dict_for_csv[word][comb]["instances"])
                        n_dict[comb_name + "_COUNT"] = counter
                        n_dict[comb_name + "%"] = counter / sum_word

                    except KeyError:
                        n_dict[comb_name + "_COUNT"] = 0
                        n_dict[comb_name + "%"] = 0
                writer.writerow(n_dict)

    """
     writes a .csv file with all sentences with the dependencies we
     are looking for
     @:param sents_csv_path: path we want to write to
     """

    def write_dict_to_csv(self, sents_csv_path):
        output_path = os.path.join(paths.files_directory,
                                   paths.dependency_set_directory,
                                   paths.dependency_set_source_files,
                                   sents_csv_path)
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
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
            self.write_all_rows_for_sentence_csv(writer)

    """
       helper function for write_dict_to_csv
       write all rows of the .csv file with the sentences we are outputting
    """

    def write_all_rows_for_sentence_csv(self, writer: csv.DictWriter):
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





if __name__ == '__main__':
    from datetime import datetime

    datetime = datetime.today().strftime('%Y_%m_%d')
    args = docopt(usage)

    file_creator = DependencySetFiles(model="en_core_web_lg", spacy_file_path=
                                       args["<spacy_file_name>"])

    csv_path = "dependency_set_from_first_{t}_posts_lg_sents_{n}.csv".format(t=args["<num_of_posts>"],
        n=datetime)
    file_creator.write_dict_to_csv(csv_path)

    counter_path = "dependency_set_from_first_{t}_posts_lg_counter_{n}.csv".format(t=args["<num_of_posts>"],
        n=datetime)

    file_creator.write_counter_csv(counter_path)









