import copy
import os.path

import spacy

import csv
from spacy.tokens import DocBin

import src.utils.path_configurations as paths

from src.source_files_by_dim.dependencies.abstract_dependency_files import \
    DependencyDimensionFiles

usage = '''
get_legal_structs CLI.
Usage:
    get_legal_structs.py
    get_legal_structs.py spacy_path csv_path as_list
'''


# TODO put correct file directories and names
# and output files

# make sure changing the doc arrangment didn't change anything else


DEPS_FOR_LIST = {"nsubj", "nsubjpass", "ccomp", "csubj"
    , "xcomp", "prt", "dobj", "prep", "dative",
                 }
# we want to find subsests of these for dependency set



class DependencyListFiles(DependencyDimensionFiles):
    # we want to find subsets of these for dependency list
    # dimension


    def __init__(self,
                 model="en_core_web_lg",
                 spacy_dir_path=r"withought_context_lg_model",
                 spacy_file_path=r"data_from_first_30000_lg_model.spacy"):


        self.DEP_LIST = {"nsubj", "nsubjpass", "ccomp", "csubj"
            , "xcomp", "prt", "dobj", "prep", "dative"}

        self.WH_WORDS = {"what", "which", "whatever", "who", "whom"
            , "whateva", "whoever", "whichever", "when", "whenever", "how"}


        self.ILLEGAL_HEADS = {"relcl", "advcl", "acl"}


        DependencyDimensionFiles.__init__(self, model=model, spacy_dir_path=
                                          spacy_dir_path, spacy_file_path=
            spacy_file_path)    # initialize super


    """
    removes dependencies we don't want in dep list
    @:param clean_punct: remove punctuation dependency from childern
    """
    def clean_token_children(self, token, clean_punct=True) -> list:
        token_children = [child for child in token.children]
        if clean_punct:
            token_children = list(filter(lambda x: x.dep_ != "punct",
                                         token_children))
        return token_children



    """
    this method checks if the token itself is of the type we want to use
    needs to be a verb and satisfy token.text.isalpha
    """
    def verify_token_type(self, token) -> bool:
        if token.pos_ != "VERB" or not token.text.isalpha():
            return False
            # get dependencies
        return True

    """
      this method checks if the token has a dependency list of the type we
      want to use.
      we pass the token_dep_list for two reasons:
      the first is better runtime,
      the second is we have cleaned the token dependencies already
    """

    def check_legal_token_deps(self, token: spacy.tokens,
                               token_dep_list: list[spacy.tokens]) -> bool:
        return self.check_if_relcl(token) and \
               self.check_if_core_dep_wh_word(token, token_dep_list) and \
        self.check_complex_core_wh_word(token, token_dep_list) and  \
        self.check_token_dep_types(token_dep_list)


    """
       this method checks whether the verb is a relcl or an acl.
       if so returns false because we do not want to use
       this type of verb
       @:param token: token to check
    """

    def check_if_relcl(self, token: spacy.tokens) -> bool:
        for child in [child for child in token.head.children]:
            if child == token and child.dep_ in self.ILLEGAL_HEADS:
                return False
        return True

    """
    check if it has a core dependent which is a wh word and preceds the
    subject
    @:param token: token to check
    @:param token_dep_list: cleaned dependencies
    """
    def check_if_core_dep_wh_word(self, token: spacy.tokens,
                                  token_dep_list: list[spacy.tokens]) -> bool:
        for dep_word in token_dep_list:
            if dep_word.dep_ in self.DEP_LIST \
                    and dep_word.i < token.i and \
                    dep_word.text in self.WH_WORDS:
                return False
        return True


    """
    Fronted-wh clauses: The verb has a core dependent which precedes the 
    subject and the dependent has a descendent (not necessarily immediate) 
    which precedes it and is a wh-word
    @:param token: token to check
    :return bool - True if there is no such a descendant
                   False otherwise
    """

    def check_complex_core_wh_word(self, token,
                                   token_dep_list: list[spacy.tokens]) \
            -> bool:
        for dep_word in token_dep_list:
            if dep_word.dep_ in self.DEP_LIST \
                    and dep_word.i < token.i:
                return not self.check_if_has_desc_which_is_wh(dep_word)
        return True


    """
    once we have made certain that there is a core dependent that precedes
    the subject, check is that dependent has a descendant that is
    a wh word
    @:param token: dependent we want to check the descendants of
    :return bool: True if there is such a descendant
                  False otherwise
    """
    def check_if_has_desc_which_is_wh(self, token: spacy.tokens) -> bool:
        filtered_subtree = set(
            filter(lambda k: k.dep_ in self.DEP_LIST and k != token,
                   [t for t in token.subtree]))
        return \
            (len(set([t.text for t in filtered_subtree]
                     ).intersection(self.WH_WORDS)) != 0)

    """
    checks that this verb has a cleaned dependency list that is a 
    subset of the dependencies we are looking for
    @:param token_dep_list: cleaned dependency list for token
    :return True if dependency list is subset
            False else
    """

    def check_token_dep_types(self, token_dep_list: list[spacy.tokens]):
        token_dep_types = [t.dep_ for t in token_dep_list]
        return set(token_dep_types).issubset(self.DEP_LIST)

    """
        arranges deps as either as list, with verb in the place that 
        it is in the sentence (with V placeholder)
        @:param token: the verb
        @token_children: cleaned token dependencies
    """

    def arrange_deps(self, token: spacy.tokens,
                     token_children: list[spacy.tokens],
                    ) -> str:

        TokenDependencies = sorted(token_children + [token], key=lambda k:
        k.i)
        TokenDependencies = [x.dep_ if x != token else "V" for
                             x in TokenDependencies]

        return "_".join(TokenDependencies)

    """
        this function is used when we are analyzing the dependency list
        we are writing a csv with the percentages and counts of the 
        different dependency lists across all verbs
    """

    def write_counter_csv(self, counter_csv_name: str):
        output_path = os.path.join(paths.files_directory,
                                   paths.dependency_list_directory,
                                   paths.dependency_list_source_files,
                                   counter_csv_name)
        with open(output_path, 'w', encoding='utf-8',
                  newline='') as f:
            clean_possible_combs = copy.deepcopy(self.possible_combs)
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
                    try:
                        n_dict[comb + "_COUNT"] = \
                            len(self.dict_for_csv[word][comb]["instances"])
                        n_dict[comb + "%"] = \
                            len(self.dict_for_csv[word][comb][
                                    "instances"]) / sum_word

                    except KeyError:
                        n_dict[comb + "_COUNT"] = 0
                        n_dict[comb + "%"] = 0
                writer.writerow(n_dict)

    """
      writes a .csv file with all sentences with the dependencies we
      are looking for
      @:param sents_csv_path: path we want to write to
    """

    def write_dict_to_csv(self, sents_csv_path):
        output_path = os.path.join(paths.files_directory,
                                   paths.dependency_list_directory,
                                   paths.dependency_list_source_files,
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
    helper function for write_dict_to_csv (in Super)
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



    file_creator = DependencyListFiles(model="en_core_web_lg",
                                       spacy_file_path=
                                       "data_from_first_50_lg_model_no_nbsp.spacy")

    csv_path = "{n}_dependency_list_from_first_50_posts_lg_sents.csv".format(
        n=datetime)
    file_creator.write_dict_to_csv(csv_path)

    counter_path = "{n}_dependency_list_from_first_25000_posts_lg_counter.csv".\
        format(
        n=datetime)

    file_creator.write_counter_csv(counter_path)



    # args = docopt.docopt(usage)
    # if args["spacy_path"] and args["csv_path"]:
    #     createWhCsv = CreateWhCsv(args["spacy_path"], args["csv_path"],
    #                               arrange_deps_as_list=args["as_list"])
    #
    # else:
    #     createWhCsv = CreateWhCsv(csv_path=os.path.join(output_dir, csv_path),
    #                               counter_csv_path=os.path.join(output_dir,
    #                                                             counter_path),
    #                               )



