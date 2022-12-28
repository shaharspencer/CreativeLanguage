import spacy
from spacy.tokens import DocBin
import csv
import pandas as pd
from docopt import docopt
import os
import spacy
from spacy.tokens import DocBin
from pathlib import Path


# load model
nlp = spacy.load('en_core_web_trf')

parts_of_speech = ["VERB","PROPN","PART","NUM","X","PUNCT","ADJ", "ADP","ADV","AUX","CCONJ", "DET", "INTJ", "NOUN", "PRON", "SCONJ","CONJ", "SYM"]

# illegal_char = {"?":"QUESTION_MARK", "#":"POUND", "%":"PERCENT", "&":"AMPERSAND", "{": "CURLY_LEFT",
#                 "}": "CURLY_RIGHT", "\":""BACK_SLASH", '"': "DOUBLE_Q", }

"""
For every word that is used as a verb on some occasion, 
Count the frequency of the word.lemma_ in different pos
And save occurences of that word in the part of speech

    Paremeters:
        spacy_path(string): a path to a .spacy file from which to create the csv
    Returns:
        for_csv(dict): 
         for every word that is a verb, for the lemma of the word,
            for every pos:
                - save sentences in which it is used as that part of speech
                - count occurences
"""

class AnalyzeVerbs:
    def __init__(self, spacy_path=
                  r"C:\Users\User\PycharmProjects\CreativeLanguage\training_data\spacy_data\data_from_first_15000_posts.spacy"):
        self.spacy_path = spacy_path
        self.verb_dict = self.analyze_verbs()
    def analyze_verbs(self):
        doc_bin = DocBin().from_disk(self.spacy_path)
        # first step: create set of all words that are used as verbs in document
        verbs = set()
        for doc in list(doc_bin.get_docs(nlp.vocab)):
            for token in doc:
                if token.pos_ != "VERB":
                    continue
                verbs.add(token.lemma_.lower())
        verbs = sorted(list(verbs))
        for_csv = {}
        for verb in verbs:
            for_csv[verb] = create_template()
        i = 0
        for doc in list(doc_bin.get_docs(nlp.vocab)):
            j = 0
            for sent in doc.sents:
                for token in sent:
                    if token.lemma_.lower() not in verbs:
                        continue
                # convert all letters but first letter to lowercase
                    for_csv[token.lemma_.lower()][token.pos_]["Instances"].add((token.text, sent.text, i, j))
                    for_csv[token.lemma_.lower()][token.pos_]["lemma"] = token.lemma_
                    for_csv[token.lemma_.lower()][token.pos_]["Counter"] += 1
                j += 1
            i += 1


        return for_csv

    def write_dict_to_csv(self, csv_path:str):
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ["word"]
            for pos in parts_of_speech:
                fieldnames.append(pos + "_lemma")
                fieldnames.append(pos + "_count")
                fieldnames.append(pos + "%")
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            d = {"word": "word"}
            for pos in parts_of_speech:
                d[pos + "_lemma"] = pos + "_lemma"
                d[pos + "_count"] = str(pos) + "_count"
                d[pos + "%"] = pos + "%"

            writer.writerow(d)
            for word in self.verb_dict.keys():
                n_dict = {'word': word}
                total = sum(
                    [self.verb_dict[word][pos]["Counter"] for pos in parts_of_speech])
                for pos in parts_of_speech:
                    n_dict[pos + "_count"] = self.verb_dict[word][pos]["Counter"]
                    n_dict[pos + "%"] = self.verb_dict[word][pos]["Counter"] / total
                    n_dict[pos + "_lemma"] = self.verb_dict[word][pos]["lemma"]

                writer.writerow(n_dict)

    def create_text_files(self):
        # forbbiden_chars = ["%", "?", "*", "'", "/", "'\'", " ", "@", "|", "!", "+","=", ":", "<", ">", "`",
        for word in self.verb_dict.keys():
            for pos in parts_of_speech:
                if not (self.verb_dict[word][pos]["Instances"]):
                    continue
                if self.verify_word(word):
                    p = os.path.join(
                        r"C:\Users\User\PycharmProjects\CreativeLanguage\explore_verb_patterns\old_python\create_verb_path_code\verb_sents",
                        word + "_" + pos + ".csv")
                    with open(p,
                              mode="w", encoding='utf-8', newline="") as f:
                        fields = ["lemma", "word form", "sentence",
                                  "doc index",
                                  "sent index"]
                        d = print_fieldnames(fields)
                        writer = csv.DictWriter(f=f, fieldnames=fields)
                        writer.writerow(d)

                        for sent in self.verb_dict[word][pos]["Instances"]:
                            verb_form, sentence, doc_index, sent_index = sent[
                                                                             0], \
                                                                         sent[
                                                                             1], \
                                                                         sent[
                                                                             2], \
                                                                         sent[
                                                                             3]
                            n_dict = {"lemma": word,
                                      "word form": verb_form,
                                      "sentence": sentence,
                                      "doc index": doc_index,
                                      "sent index": sent_index}

                            writer.writerow(n_dict)

    def verify_word(self, word: str)->bool:
        illegal = [':', '*', "?", "<", ">", "|", '"', chr(92), chr(47)]
        return not any(ill in word for ill in illegal)

"""
create template for dictionary for analyze_verbs function
    Parameters: None
    return: dict(dict) : dictionary for every word
    containe
"""
def create_template():
    dict = {}
    for pos in parts_of_speech:
        dict[pos] = {"lemma": "", "Counter":0, "Instances":set()}
    return dict


def print_fieldnames(given_lst: iter):
    dic = {}
    for fieldname in given_lst:
        dic[fieldname] = fieldname
    return dic





if __name__ == '__main__':
    verb_anazlyzer = AnalyzeVerbs()

    verb_anazlyzer.write_dict_to_csv("verb_path_new.csv")

    verb_anazlyzer.create_text_files()
