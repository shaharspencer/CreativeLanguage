"""
this program chooses k sentences from the saved spacy file
and chooses for each sentence a random verb
"""
import csv
import os
import random

import spacy
from docopt import docopt

# import utils.path_configurations as paths
from spacy.tokens import DocBin

usage = '''
randomize_sentences CLI.
Usage:
    randomize_sentences.py <k>
'''


class RandomizeSents:
    """
    k is the number of sentences

    """
    def __init__(self, k: int, output_path,
                 model="en_core_web_lg",
                 spacy_directory=
                 r"withought_context_lg_model",
                 ):
        # get full spacy path
        # self.spacy_path = os.path.join(paths.files_directory,
        #                                paths.spacy_files_directory,
        #                                spacy_directory,
        #                                spacy_file_path)

        self.spacy_path = r"C:\Users\User\OneDrive\Documents\CreativeLanguageOutputFiles\training_data\spacy_data\withought_context_lg_model\data_from_first_40000_lg_model_spacy_3.5.5.spacy"

        self.output_path = output_path
        self.k = k

        # open docBin with saved spacy corpus
        self.doc_bin = DocBin().from_disk(self.spacy_path)

        # initialize nlp model
        self.nlp = spacy.load(model)

        self.doc_list = list(self.doc_bin.get_docs(self.nlp.vocab))

    """
    creates the dictionary for the csv
    and writes to the csv
    """
    def create_dict_and_write_to_csv(self):
        self.__create_k_sents_dict()
        self.__get_k_sents_csv()

    """
    creates the dictionary for the csv
    """
    def __create_k_sents_dict(self):
        self.sents_mapped_to_verbs = {}
        indexes = self.__randomize_k_indexes()

        for index in indexes:
            doc_at_index = self.doc_list[index]
            # choose verb
            verbs_in_doc = [token for token in doc_at_index if
                            token.pos_ ==
                            "VERB"]

            rand_int = index
            alt_indexes = indexes.copy()
            alt_indexes.remove(index)
            while len(verbs_in_doc) == 0 or rand_int in alt_indexes:

               rand_int = random.choice(range(len(self.doc_list)))
               doc_at_index = self.doc_list[rand_int]
               # choose verb
               verbs_in_doc = [token for token in doc_at_index if
                               token.pos_ ==
                               "VERB"]

            # choose verb
            verb_choice = random.choice(verbs_in_doc)
            self.sents_mapped_to_verbs[doc_at_index] = verb_choice


    """
    creates the sentence csv
    """
    def __get_k_sents_csv(self):

        with open(self.output_path, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ["sentence", "verb", "dep set", "doc index", "sent index"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writerow(self.__fieldnames_for_csv(fieldnames))

            for key in self.sents_mapped_to_verbs.keys():
                dep_set = set([child.dep_ for child in self.sents_mapped_to_verbs[key].children])
                if "punct" in dep_set:
                    dep_set.remove("punct")
                dep_set = "_".join(dep_set)
                n_dict = {"sentence": key.text,
                          "verb": self.sents_mapped_to_verbs[key].text,
                          "doc index": key.user_data["DOC_INDEX"],
                          "sent index":
                          key.user_data["SENT_INDEX"], "dep set": dep_set}
                writer.writerow(n_dict)




    """
    this method chooses k indexes from the length of the corpus (in sentences)
    """
    def __randomize_k_indexes(self) -> list[int]:
        randomize_from_len = len(self.doc_list)
        return random.sample(range(0, randomize_from_len), self.k)


    """
    get fieldnames for csv in dict format
    """
    def __fieldnames_for_csv(self, given_lst: iter):
        dic = {}
        for fieldname in given_lst:
            dic[fieldname] = fieldname
        return dic



if __name__ == '__main__':

    args = docopt(usage)
    # spacy_file_path = args["<spacy_file_name>"]

    k = int(args["<k>"])

    output_file_name = "random_sents.csv"

    randomizer = RandomizeSents(k=k, output_path=output_file_name)
    randomizer.create_dict_and_write_to_csv()