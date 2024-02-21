import conllu.models

from src.masking_subproject.NER.base_functions import get_spacy_ners_from_conllu_sent, get_nlp, get_gold_ner, load_data, get_spacy_ners_from_list_sent, NER_MAP
from src.masking_subproject.tagging.tag_with_mask import FillMask

fill_masks = {"1": FillMask(top_k=1), "3": FillMask(top_k=3)}
def replace_tokens(tokens:list[str], start_index:int, end_index: int, replecament_token: str):
    masked_tokens = tokens[:start_index] + [replecament_token] + tokens[end_index:]
    return masked_tokens

nlp = get_nlp()

def get_named_entity_after_masking(sent: conllu.models.TokenList, start_index:int, end_index:int, k:int):
    res = []
    predictions = []
    fill_mask = fill_masks[str(k)]
    sent_text_list = [str(w) for w in sent]
    masked_sentence = replace_tokens(sent_text_list, start_index=start_index, end_index=end_index, replecament_token="<mask>")
    sent_text = " ".join(masked_sentence)
    fillmask_res = fill_mask.replace_token(sent_text)
    for r in fillmask_res:
        resulted_sentence = replace_tokens(tokens=sent_text_list, start_index=start_index, end_index=end_index, replecament_token=r["token_str"])
        original_span_length = end_index - start_index
        joint_sent, flag = " ".join(resulted_sentence), False
        sent_res = nlp(joint_sent)
        replacement_length = len(r["token_str"].split())
        len_diff = replacement_length - original_span_length
        new_end_index = end_index + len_diff

        for ent in sent_res.ents:
            if ent.start == start_index and ent.end == new_end_index and ent.label_ in NER_MAP.keys():
                z = 0
                predictions.append(ent.label_)
                flag = True
        if not flag:
            predictions.append("O")



    return predictions


        # add data to results file
#TODO maybe can change the condition here. can check if we have predicted an overall correct tag for the token range. ex. part of a location, ex. part of a name. need to give some thoght
def predict_on_spacy_mistakes(data):
    correct_preds = 0
    masked_ners = {}
    for sent in data:
        spacy_ners = get_spacy_ners_from_conllu_sent(sent)
        gold_ners = get_gold_ner(sent)
        # see which ners spacy did not identify correctly, and mask them and retag
        for item in spacy_ners:

            if not item in gold_ners: #TODO need to somehow put in case where we get two seperate predictions
                preds = get_named_entity_after_masking(sent, start_index=item["start_index"], end_index=item["end_index"], k=3)
                max_pred = max(set(preds), key = preds.count)
                flag = True
                for gold in gold_ners:
                    if gold["start_index"] == item["start_index"] and gold["end_index"] == item["end_index"]:
                        flag = False
                        if gold["label"] == max_pred:
                            correct_preds += 1
                if flag and max_pred == "O":
                    correct_preds += 1




if __name__ == '__main__':
    data = load_data(r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\NER\raw_data\en_ewt-ud-test.conllu")
    predict_on_spacy_mistakes(data)