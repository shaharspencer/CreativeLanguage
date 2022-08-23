import spacy
from spacy.tokens import DocBin
from pathlib import Path
from collections import Counter
import operator
import csv
import pandas as pd
from docopt import docopt


nlp = spacy.load('en_core_web_trf')

"""
Function mean to test if the .spacy file has properly saved parts of speech 
and dependencies in text.

    Paremeters:
        None
    Returns: 
        None
"""
def load_nlp_file():
    doc_bin = DocBin().from_disk("data_from_first_1000_posts.spacy")
    i = 0
    for doc in list(doc_bin.get_docs(nlp.vocab)):
        with open("ent_pos_post/ent_pos_"+str(i), "w") as f:
            for ent in doc:
                f.write(str(ent)+ " " + str(ent.pos_))
                f.write("\n")

        for sent in doc.sents:
            svg = spacy.displacy.render(sent, style='dep', jupyter=False)
            output_path = Path(
                "data_vis_post/data_vis_" + str(
                    i) + ".svg")
            output_path.open("w", encoding="utf-8").write(svg)
            break