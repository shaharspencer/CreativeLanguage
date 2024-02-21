import json
from typing import Tuple, List, Dict, Any
from base_functions import get_spacy_ners_from_conllu_sent, get_gold_ner, load_data
import pandas as pd
import spacy
from conllu import parse
import conllu
from spacy import tokens
from spacy.tokens import Doc
from sklearn.metrics import accuracy_score




def get_spacy_measures(data):
    ners = {}
    for sent_index, sent in enumerate(data):
        ners[sent.metadata["sent_id"]] = {"spacy_default_tags": get_spacy_ners_from_conllu_sent(sent), "gold_tags": get_gold_ner(sent)}
        print(sent_index)
    with open("ner_results.json", "w") as outfile:
        json.dump(ners, outfile)





if __name__ == '__main__':
    data = load_data(r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\NER\raw_data\en_ewt-ud-test.conllu")
    get_spacy_measures(data)