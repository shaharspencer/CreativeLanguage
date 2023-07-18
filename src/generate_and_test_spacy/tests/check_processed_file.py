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

    doc_bin_2 = DocBin().from_disk(r"C:\Users\User\PycharmProjects\CreativeLanguageWithVenv\src\generate_and_test_spacy\processors\data_from_first_2_lg_model_spacy_3.5.5.spacy")

    x = 0

    for doc in doc_bin_2.get_docs(nlp.vocab):

        # with open("ent_pos_post/ent_pos_"+str(i), "w") as f:
        #     for ent in doc:
        #         f.write(str(ent)+ " " + str(ent.pos_))
        #         f.write("\n")

        for sent in doc.sents:
            cor = [[token, token.pos_] for token in sent]
            print(cor)
            print("\n")
            svg = spacy.displacy.render(sent, style='dep', jupyter=False)
            # output_path = Path(
            #     "data_vis_post_2/data_vis_" + str(
            #         x) + ".svg")
            # output_path.open("w", encoding="utf-8").write(svg)
            # break
        x += 1
        print(x)
        print(sent.text)
        if x == 5:
            break




if __name__ == '__main__':
    load_nlp_file()