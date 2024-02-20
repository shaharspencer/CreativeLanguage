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


    total_correct = 0
    total_predictions_for_precision, total_predictions_for_recall = 0, 0

    for sent_id, sentence_data in ners.items():
        spacy_entities = set((ent['text'], ent['label'], ent["start_index"], ent["end_index"]) for ent in sentence_data['spacy_default_tags'])
        gold_entities = set((ent['text'], ent['label'], ent["start_index"], ent["end_index"]) for ent in sentence_data['gold'])

        total_correct += len(spacy_entities.intersection(gold_entities))
        total_predictions_for_precision += len(spacy_entities)
        total_predictions_for_recall += len(gold_entities)
        # if the preictions did not match see why


    overall_precision = total_correct / total_predictions_for_precision if total_predictions_for_precision > 0 else 0
    overall_recall = total_correct / total_predictions_for_recall if total_predictions_for_recall > 0 else 0
    print("spacy baseline results")
    print(f"precision: {overall_precision}")
    print(f"recall: {overall_recall}")







if __name__ == '__main__':
    data = load_data(r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\NER\raw_data\en_ewt-ud-test.conllu")
    get_spacy_measures(data)