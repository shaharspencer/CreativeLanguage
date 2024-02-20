from typing import Tuple, List, Dict, Any

import pandas as pd
import spacy
from conllu import parse
import conllu
from spacy import tokens
from spacy.tokens import Doc
from sklearn.metrics import accuracy_score

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        words = [word for word in words if word.strip()]  # Remove empty tokens
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)
def get_nlp():
    nlp = spacy.load('en_core_web_lg')
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    return nlp
#Person (PER)
# Organization (ORG)
# Location (LOC)
NER_MAP = {
"GPE": "LOC",
"LOC": "LOC",
"ORG": "ORG",
"PERSON": "PER"}

#TODO case where we have a ner that wasnt recognized by spacy
def load_data(raw_data_file):
    with open(raw_data_file, 'r', encoding='utf-8') as conllu_file:
        conllu_content = parse(conllu_file.read())
    return conllu_content

nlp = get_nlp()
def get_spacy_ners_from_conllu_sent(sent: conllu.models.TokenList)-> list[dict[str, Any]]:
    spacy_ners = []
    sent_text = " ".join([str(w) for w in sent])
    doc = nlp(sent_text)
    for ent in doc.ents:
        ent_text, ent_label = ent.text, ent.label_
        # add conversion with ner map
        if ent_label in NER_MAP.keys():
            ent_label = NER_MAP[ent_label]

            d = {"text": ent_text, "label": ent_label, "start_index": ent.start, "end_index": ent.end, "sent_id": sent.metadata["sent_id"]}
            spacy_ners.append(d)
        else:
            # if not in ner map we dont have data
            pass

    return spacy_ners

def get_spacy_ners_from_list_sent(sent: list[str], sent_id)-> list[dict[str, Any]]:
    spacy_ners = []
    sent_text = " ".join([str(w) for w in sent])
    doc = nlp(sent_text)
    for ent in doc.ents:
        ent_text, ent_label = ent.text, ent.label_
        # add conversion with ner map
        if ent_label in NER_MAP.keys():
            ent_label = NER_MAP[ent_label]

            d = {"text": ent_text, "label": ent_label, "start_index": ent.start, "end_index": ent.end, "sent_id": sent_id}
            spacy_ners.append(d)
        else:
            # if not in ner map we dont have data
            pass

    return spacy_ners



# def get_gold_label(sent: conllu.models.TokenList , ner_dict: dict)-> dict:
#     #todo case where there are two seperate labels for the ner for some reason
#     #todo remove prefix of I- or B-
#     b, e = ner_dict["begin_index"], ner_dict["end_index"]
#     t = [(sent[i]["id"]-1, sent[i]["form"], sent[i]["lemma"]) for i in range(b, e)]
#     rest_of_sent = e
#     #todo maybe bug here in condition rest_of_sent < len(sent)
#     while rest_of_sent < len(sent) and sent[rest_of_sent]["lemma"].startswith("I-"):
#         t.append(((sent[rest_of_sent]["id"]-1, sent[rest_of_sent]["form"], sent[rest_of_sent]["lemma"])))
#         rest_of_sent += 1
#     gold_label_dict = {"text": " ".join(m[1] for m in t), "label":
#         set([m[2][2:] if (m[2].startswith("I-") or m[2].startswith("B-") ) else m[2] for m in t])
#         , "start_index": b, "end_index": rest_of_sent}
#     return gold_label_dict


def get_gold_ner(sent: conllu.models.TokenList):
    named_entities = []

    current_entity = None
    current_entity_type = None
    start_index = None

    for token in sent:
        if token['lemma'].startswith("B-"):
            if current_entity:
                named_entities.append({
                    'text': current_entity,
                    'label': current_entity_type,
                    'start_index': start_index['id'] - 1,
                    'end_index': token['id'] - 1, "sent_id": sent.metadata["sent_id"]
                })
            current_entity = token['form']
            current_entity_type = token['lemma'][2:]
            start_index = token
        elif token['lemma'].startswith("I-"):
            if current_entity_type == token['lemma'][2:]:
                current_entity += " " + token['form']
            else:
                print("Error: Mismatched entity types.")
                continue
        else:
            if current_entity:
                named_entities.append({
                    'text': current_entity,
                    'label': current_entity_type,
                    'start_index': start_index['id'] - 1,
                    'end_index': token['id'] - 1, "sent_id": sent.metadata["sent_id"]
                })
                current_entity = None
                current_entity_type = None

    # Handling the last entity if it exists
    if current_entity:
        named_entities.append({
            'text': current_entity,
            'label': current_entity_type,
            'start_index': start_index['id'] - 1,
            'end_index': sent[-1]['id'], "sent_id": sent.metadata["sent_id"]
        })

    return named_entities




