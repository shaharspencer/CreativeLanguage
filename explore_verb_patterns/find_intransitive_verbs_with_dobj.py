import spacy
from spacy.tokens import DocBin
import csv
import pandas as pd
from docopt import docopt
import os
import spacy
from spacy.tokens import DocBin
from pathlib import Path



# maybe use matcher


# load model
nlp = spacy.load('en_core_web_trf')

parts_of_speech = ["VERB","PROPN","PART","NUM","X","PUNCT","ADJ", "ADP","ADV","AUX","CCONJ", "DET", "INTJ", "NOUN", "PRON", "SCONJ","CONJ", "SYM"]




def create_dep_csv(spacy_path, csv_path):
    doc_bin = DocBin().from_disk(spacy_path)

    verb_dict = {}
    for doc in list(doc_bin.get_docs(nlp.vocab)):
        for token in doc:
            if token.is_punct or token.is_space or token.is_stop:
                continue
            if token.pos_ != "VERB":
                continue
            if token.lemma_ in verb_dict:
                verb_dict[token.lemma_]["count"] += 1
                for t in token.children:
                    if t.dep_ == "dobj":
                        verb_dict[token.lemma_]["has_dobj"] += 1
            else:
                verb_dict[token.lemma_] = {"count":0, "has_dobj":0}
                verb_dict[token.lemma_]["count"] += 1
                for t in token.children:
                    if t.dep_ == "dobj":
                        verb_dict[token.lemma_]["has_dobj"] += 1
                        break

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ["word", "count", "has_dobj"]
        with open(csv_path, 'w', encoding='utf-8') as f:

            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writerow({'word': "word", 'count':"count", "has_dobj":"has_dobj"})
            for word in verb_dict:
                writer.writerow({'word': word, 'count': verb_dict[word]["count"], 'has_dobj':
                                 verb_dict[word]["has_dobj"]})

def print_interesting(spacy_path):
    doc_bin = DocBin().from_disk(spacy_path)
    df = pd.read_csv("has_dobj.csv")
    print(df.loc[df['word'] == "sneeze"])
    txt = open("sents.txt", "w")
    for doc in list(doc_bin.get_docs(nlp.vocab)):
        for token in doc:
            if token.is_punct or token.is_space or token.is_stop:
                continue
            if token.pos_ != "VERB":
                continue

            try:
                row = df.loc[df['word'] == token.lemma_]
                if "dobj" in [c.dep_ for c in token.children] and float(row["has_dobj"]) / int(row["count"]) < 0.02:
                    print(token.text+":")
                    txt.write(token.text+": ")
                    txt.write(str(token.sent))
                    print(token.sent)
            except KeyError:
                continue






if __name__ == '__main__':
    dict = create_dep_csv(
        "../training_data/spacy_data/data_from_first_15000_posts.spacy", "has_dobj.csv")
    # write_dict_to_csv(dict,"verbed.csv")
    print_interesting(
        "../training_data/spacy_data/data_from_first_15000_posts.spacy")