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

nlp = spacy.load('en_core_web_lg')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
#Person (PER)
# Organization (ORG)
# Location (LOC)
NER_MAP = {
"GPE": "LOC",
"LOC": "LOC",
"ORG": "ORG",
"PERSON": "PER",
"PRODUCT": "todo"}

#TODO case where we have a ner that wasnt recognized by spacy
def ner_with_spacy(sent: conllu.models.TokenList)-> dict:
    sent_ners = []
    sent_text = " ".join([str(w) for w in sent])
    doc = nlp(sent_text)
    for ent in doc.ents:
        ent_text, ent_label = ent.text, ent.label_
        # add conversion with ner map
        if ent_label in NER_MAP.keys():
            ent_label = NER_MAP[ent_label]

            d = {"text": ent_text, "label": ent_label, "begin_index": ent.start, "end_index": ent.end}
            sent_ners.append(d)
        else:
            # if not in ner map we dont have data
            pass

    ners = {sent.metadata["sent_id"]: sent_ners}
    return ners




def get_gold_ners(sent: conllu.models.TokenList)-> dict:
    #todo case where there are two seperate labels for the ner for some reason
    #todo remove prefix of I- or B-
    b, e = ner_dict["begin_index"], ner_dict["end_index"]
    t = [(sent[i]["id"]-1, sent[i]["form"], sent[i]["lemma"]) for i in range(b, e)]
    rest_of_sent = e
    #todo maybe bug here in condition rest_of_sent < len(sent)
    while rest_of_sent < len(sent) and sent[rest_of_sent]["lemma"].startswith("I-"):
        t.append(((sent[rest_of_sent]["id"]-1, sent[rest_of_sent]["form"], sent[rest_of_sent]["lemma"])))
        rest_of_sent += 1
    gold_label_dict = {"text": " ".join(m[1] for m in t), "label":
        set([m[2][2:] if (m[2].startswith("I-") or m[2].startswith("B-") ) else m[2] for m in t])
        , "start_index": b, "end_index": rest_of_sent}
    return gold_label_dict



def get_spacy_accuracy(data):
    all_ners = {}
    for sent in data:
        ners = ner_with_spacy(sent)
        all_ners.update(ners)

    # next flatten structure
    flat_ners = []
    df = pd.DataFrame(columns=["spacy_text", "spacy_label", "gold_standard_text", "gold_standard_label"])
    for k, v in all_ners.items():
        for dict_pair in v:
            df = df._append({"spacy_text": dict_pair["spacy_tag"]["text"],
                       "spacy_label": dict_pair["spacy_tag"]["label"],
                       "gold_standard_text": dict_pair["gold_label"]["text"],
                       "gold_standard_label": list(dict_pair["gold_label"]["label"])}, ignore_index=True)



    # now get accuracy somehow
    accuracy = accuracy_score(y_true=df["gold_standard_label"], y_pred=df["spacy_label"])



def load_data(raw_data_file):
    with open(raw_data_file, 'r', encoding='utf-8') as conllu_file:
        conllu_content = parse(conllu_file.read())
    return conllu_content


if __name__ == '__main__':
    data = load_data(r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\NER\raw_data\en_ewt-ud-test.conllu")
    get_spacy_accuracy(data)