import spacy
from spacy.tokens import DocBin
import csv
import pandas as pd
from docopt import docopt
import os


# load model
nlp = spacy.load('en_core_web_trf')

parts_of_speech = ["VERB","PROPN","PART","NUM","X","PUNCT","ADJ", "ADP","ADV","AUX","CCONJ", "DET", "INTJ", "NOUN", "PRON", "SCONJ","CONJ", "SYM"]

# illegal_char = {"?":"QUESTION_MARK", "#":"POUND", "%":"PERCENT", "&":"AMPERSAND", "{": "CURLY_LEFT",
#                 "}": "CURLY_RIGHT", "\":""BACK_SLASH", '"': "DOUBLE_Q", }

# text = "he gave her a knowing look. she had a traumatizing accident. the event was escalating."
# t = nlp(text)
# for token in nlp(text):
#     print((token.text, token.lemma_))
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
def analyze_verbs(spacy_path):
    doc_bin = DocBin().from_disk(spacy_path)
    # first step: create set of all words that are used as verbs in document
    verbs = set()
    for doc in list(doc_bin.get_docs(nlp.vocab)):
        for token in doc:
            if token.pos_ != "VERB":
                continue
            verbs.add(token.text.lower())
    verbs = sorted(list(verbs))
    for_csv = {}
    for verb in verbs:
        for_csv[verb] = create_template()
    for doc in list(doc_bin.get_docs(nlp.vocab)):
            for sent in doc.sents:
                for token in sent:
                    if token.text.lower() not in verbs:
                        continue
                # convert all letters but first letter to lowercase
                    for_csv[token.text.lower()][token.pos_]["Instances"].add(sent.text)
                    for_csv[token.text.lower()][token.pos_]["lemma"] = token.lemma_
                    for_csv[token.text.lower()][token.pos_]["Counter"] += 1

    return for_csv

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


def write_dict_to_csv(dict, csv_path):
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ["word"]
        for pos in parts_of_speech:
            fieldnames.append(pos+"_lemma")
            fieldnames.append(pos+"_count")
            fieldnames.append(pos+"%")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        d = {"word":"word"}
        for pos in parts_of_speech:
            d[pos+"_lemma"] = pos+"_lemma"
            d[pos+"_count"] = str(pos)+"_count"
            d[pos+"%"] = pos+"%"
        writer.writerow(d)
        for word in dict.keys():
            n_dict = {'word':word}
            total = sum([dict[word][pos]["Counter"] for pos in parts_of_speech])
            for pos in parts_of_speech:
                n_dict[pos+"_count"] = dict[word][pos]["Counter"]
                n_dict[pos+"%"] = dict[word][pos]["Counter"] / total
                n_dict[pos+"_lemma"] = dict[word][pos]["lemma"]
            writer.writerow(n_dict)


# def encode(m):
#     encoded = [c for c in m]
#     for i, c in enumerate(m):
#         try:
#             encoded[i] = illegal_char[c]
#         except KeyError:
#             pass
#     return ''.join(encoded)
#

def create_text_files(dict):
    forbbiden_chars = ["%", "?", "*", "'", "/", "'\'", " ", "@", "|", "!", "+","=", ":", "<", ">", "`"]
    for word in dict.keys():
        for pos in parts_of_speech:
            if any(ele in word for ele in forbbiden_chars):
                if os.path.exists(r"verb_files/weird_words"):
                    append_write = 'a'  # append if already exists
                else:
                    append_write = 'w'
                with open(r"verb_files/weird_words" + pos, mode=append_write, encoding='utf-8') as f:
                    for sent in dict[word][pos]["Instances"]:
                        f.write(sent)
                        f.write("\n")
                        continue

                    with open(r"verb_files/weird_words" + pos, mode=append_write, encoding='utf-8') as f:
                        for sent in dict[word][pos]["Instances"]:
                            f.write(sent)
                            f.write("\n")
            else:
                with open(r"verb_files/" + pos + "_"+ word,
                          mode="w", encoding='utf-8') as f:
                    for sent in dict[word][pos]["Instances"]:
                        f.write(sent)
                        f.write("\n")



if __name__ == '__main__':
    dict = analyze_verbs("./data_from_first_1000_posts.spacy")
    # write_dict_to_csv(dict,"verb_path.csv")
    # create_text_files(dict)