import enum
import os.path
from spacy.tokens import DocBin
from abc import ABC, abstractmethod
from src.utils import path_configurations as paths

import spacy

from spacy.tokens import DocBin

import csv

import os


class DEP_MODE(enum.Enum):
    SINGLE_DEP = 1
    DEP_GROUP = 2


class DEP_SET_TYPE(enum.Enum):
    COMPLETE_COMPELEMENTS = {"ccomp", "xcomp",
                             "prt", "dobj", "prep", "dative"}
    NON_CLAUSAL_COMPELEMENTS = {"prt", "dobj", "prep", "dative"}


class DependencyDimensionFiles(ABC):

    def __init__(self, spacy_dir_path, spacy_file_path,
                 model, mode:DEP_MODE):
        self.mode = mode
        self.nlp = spacy.load(model)

        # get saved spacy file
        self.spacy_path = os.path.join(paths.files_directory,
                                       paths.spacy_files_directory,
                                       spacy_dir_path, spacy_file_path)

        # get docbin from spacy path
        self.doc_bin = DocBin().from_disk(self.spacy_path)

        self.initialize_run()

    """
    starts the creation dictionary that we will write to the .csv files
    """

    def initialize_run(self):
        # save all possible combinations (that we find while running)
        self.possible_combs = set()

        # create dictionary
        self.dict_for_csv = {}
        self.create_csv_dict()

    """
    actually creates dictionary for the files
    """

    def create_csv_dict(self):
        for doc in self.doc_bin.get_docs(self.nlp.vocab):
            self.add_doc_to_dict(doc)


    """
    adds a single doc object to dictionary
    each doc should only have one sentence
    """

    def add_doc_to_dict(self, doc):
        for sent in doc.sents:
            for token in sent:
                if self.mode == DEP_MODE.SINGLE_DEP:
                    self.add_token_to_dict_by_single_dep(doc.user_data["DOC_INDEX"],
                                                 doc.user_data["SENT_INDEX"], token)

                elif self.mode == DEP_MODE.DEP_GROUP:
                    self.add_token_to_dict_by_subset(doc.user_data["DOC_INDEX"],
                                                 doc.user_data["SENT_INDEX"], token)

                else:
                    raise Exception("the mode in the abstract file was wrong")

    """
        adds a single token to dictionary
        first checks the token is the type we want to add
        (is a verb, checks the deps are the type we want)
        then verifys token has a key in the dictionary
        then adds to dictionary
    """

    def add_token_to_dict_by_subset(self, doc_index: int, sent_indx: int,
                                    token: spacy.tokens) -> bool:

        token_children = self.clean_token_children(token)
        # verify that we want to add this token
        if not self.check_if_add_token(token, token_children):
            return False
        # create key for token if not yet in dict keys
        self.add_verb_template_to_dict(token, dep_list=token_children)
        # arrange dependencies
        token_dep_comb = self.arrange_deps(token, token_children)

        self.add_entry_to_dict(token, doc_index, sent_indx, token_dep_comb, token_children)

    def add_token_to_dict_by_single_dep(self, doc_index: int, sent_indx: int,
                                    token: spacy.tokens) -> bool:

        token_children = self.clean_token_children(token)
        # verify that we want to add this token
        if not self.check_if_add_token(token, token_children):
            return False
        # create key for token if not yet in dict keys
        self.add_verb_template_to_dict(token, )
        # arrange dependencies
        # token_dep_comb = self.arrange_deps(token, token_children)
        for dep in token_children:
            self.add_entry_to_dict(token, doc_index, sent_indx, dep.dep_,
                                   token_children)


    def add_entry_to_dict(self, token, doc_index, sent_indx, token_dep_comb,
                          token_children:list[spacy.tokens]):
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
    this method checks if the token itself is of the type we want to use
    needs to be a verb and satisfy token.text.isalpha
    """

    @abstractmethod
    def verify_token_type(self, token) -> bool:
        pass

    """
    checks that the dependencies are of the type we want
    """

    @abstractmethod
    def check_legal_token_deps(self, token: spacy.tokens,
                               token_dep_list: list[spacy.tokens]) -> bool:
        pass

    """
      checks:
      that token is of the type we want to add
      that token dependencies are of the type we want
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
    arranges the token dependencies in a specific order, either with
    or without verb
    """
    @abstractmethod
    def arrange_deps(self, token: spacy.tokens,
                     token_children: list[spacy.tokens],
                     ) -> str:
        pass


    """
    writes a .csv file with all sentences with the dependencies we
    are looking for
    @:param sents_csv_path: path we want to write to
    """
    @abstractmethod
    def write_dict_to_csv(self, sents_csv_path):
       pass

    """
    helper function for write_dict_to_csv
    write all rows of the .csv file with the sentences we are outputting
    """
    @abstractmethod
    def write_all_rows_for_sentence_csv(self, writer: csv.DictWriter):
        pass

    """
    removes dependencies we don't want in dep list
    @:param clean_punct: remove punctuation dependency from childern
    """

    @abstractmethod
    def clean_token_children(self, token, clean_punct=True) -> list:
        pass

    """
       this function is used when we are analyzing the dependencies
       we are writing a csv with the percentages and counts of the 
       different dependency groups (set of list) across all verbs
    """

    @abstractmethod
    def write_counter_csv(self, counter_csv_path: str, column_set: list[str]):
        pass

    """
    each token needs a key in the final dictionary
    if the token.lemma_lower() is not yet a key, add as key
    """

    def add_verb_template_to_dict(self, token: spacy.tokens,
                                  dep_list:list[str]):
        if not token.lemma_.lower() in self.dict_for_csv.keys():
            self.dict_for_csv[token.lemma_.lower()] = {}

    """
    creates a dictionary for the .csv counter file fieldnames
    """
    def print_fieldnames(self, comb: iter, column_set):
        dic = {}
        for fieldname in comb:
            if "COUNT" in column_set:
                dic[fieldname + "_COUNT"] = fieldname + "_COUNT"
            if "%" in column_set:
                dic[fieldname + "%"] = fieldname + "%"
        return dic



