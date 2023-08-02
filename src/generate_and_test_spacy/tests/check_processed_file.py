import os.path

import spacy
# print(spacy.__version__)
from spacy.tokens import DocBin
from pathlib import Path
from collections import Counter
import operator
import csv
import pandas as pd
from docopt import docopt


nlp = spacy.load('en_core_web_lg')
# print(nlp.meta)
#
"""
Function mean to test if the .spacy file has properly saved parts of speech 
and dependencies in text.

    Paremeters:
        None
    Returns: 
        None
"""
def load_nlp_file(file_directory = r"C:\Users\User\OneDrive\Documents\CreativeLanguageOutputFiles\training_data\spacy_data\withought_context_lg_model",
                 file_name_1 = "data_from_first_5_lg_model_spacy_3.5.5.spacy",
                  file_name_2 = "data_from_first_40000_lg_model_spacy_3.5.5.spacy"):
    # file_path_1 = os.path.join(file_directory, file_name_1)
    # file_path_2 = os.path.join(file_directory, file_name_2)
    doc_bin_1 = DocBin().from_disk(r"C:\Users\User\PycharmProjects\CreativeLanguageWithVenv\src\generate_and_test_spacy\tests\7_30_2023_data_from_first_20_lg_model_spacy_3.5.5.spacy")
    doc_bin_2 = DocBin().from_disk(r"CPU_7_30_2023_data_from_first_{n}_lg_model_spacy_3.5.5."
                                "spacy")


    for doc_1, doc_2 in zip(doc_bin_1.get_docs(nlp.vocab), doc_bin_2.get_docs(nlp.vocab)):
        for sent_1, sent_2 in zip(doc_1.sents, doc_2.sents):
            print(f"sentence is: {sent_1}\n")
            cor_1 = [[token, token.pos_] for token in sent_1]
            cor_2 = [[token, token.pos_] for token in sent_2]
            lists_equal = are_lists_equal(cor_1, cor_2)
            print(f"predictions are identical: {lists_equal}\n")
            assert lists_equal
            print(f"gpu preds: {cor_1}\n")
            print(f"cpu preds: {cor_2}")
            print("\n")

def are_lists_equal(list1, list2):
    if len(list1) != len(list2):
        return False

    for i in range(len(list1)):
        for j in range(len(list1[i])):
            if j == 0 and list1[i][j].text != list2[i][j].text:
                return False

            if j == 1 and list1[i][j] != list2[i][j]:
                return False

    return True




if __name__ == '__main__':
    load_nlp_file()