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
    i = 0
    for doc in list(doc_bin.get_docs(nlp.vocab)):
        j = 0
        for sent in doc.sents:
            for token in sent:
                if token.text.lower() not in verbs:
                    continue
            # convert all letters but first letter to lowercase
                for_csv[token.text.lower()][token.pos_]["Instances"].add((sent.text, i, j))
                for_csv[token.text.lower()][token.pos_]["lemma"] = token.lemma_
                for_csv[token.text.lower()][token.pos_]["Counter"] += 1
            j += 1
        i += 1


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
    # dataFrame = pd.read_csv(csv_path)
    # dataFrame = dataFrame.sort_values("VERB_count", axis=0, ascending=True, inplace=True,na_position='first',)



def create_text_files(dict):
    # forbbiden_chars = ["%", "?", "*", "'", "/", "'\'", " ", "@", "|", "!", "+","=", ":", "<", ">", "`",
    for word in dict.keys():
        for pos in parts_of_speech:
            if not (dict[word][pos]["Instances"]):
                continue
            if any(not (ele.islower() or ele.isupper()) for ele in word):
                if os.path.exists(r"C:\Users\User\PycharmProjects\indexing_text\verb_sents/weird_words_"+pos+".txt"):
                    append_write = 'a'  # append if already exists
                else:
                    append_write = 'w'
                with open(r"C:\Users\User\PycharmProjects\indexing_text\verb_sents\weird_words_" + pos+".txt", mode=append_write, encoding='utf-8') as f:
                    for sent in dict[word][pos]["Instances"]:
                        f.write("doc num " + str(sent[1]) + ", sent num " + str(sent[2])+",")
                        f.write("[")
                        f.write(word)
                        f.write("]: ")
                        f.write(sent[0])
                        f.write("\n")
            else:
                with open(r"C:\Users\User\PycharmProjects\indexing_text\verb_sents/" + word + "_"+ pos +".txt",
                          mode="w", encoding='utf-8') as f:
                    for sent in dict[word][pos]["Instances"]:
                        f.write("doc num " + str(sent[1]) + ", sent num " + str(sent[2])+",")
                        f.write(sent[0])
                        f.write("\n")


if __name__ == '__main__':
    dict = analyze_verbs(r"C:\Users\User\PycharmProjects\indexing_text\spacy_data\data_from_first_15000_posts.spacy")
    write_dict_to_csv(dict,"verb_path_new.csv")

    create_text_files(dict)
    pass