import conllu.models


from src.masking_subproject.NER.base_functions import get_spacy_ners_from_conllu_sent, get_nlp, get_gold_ner, load_data, get_spacy_ners_from_list_sent, NER_MAP
from src.masking_subproject.tagging.tag_with_mask import FillMask

fill_masks = {"1": FillMask(top_k=1),}
              # "9": FillMask(top_k=9)}
def replace_tokens(tokens:list[str], start_index:int, end_index: int, replecament_token: str):
    masked_tokens = tokens[:start_index] + [replecament_token] + tokens[end_index:]
    return masked_tokens

nlp = get_nlp()

def get_named_entity_after_masking(sent: conllu.models.TokenList, start_index:int, end_index:int, k:int):
    res = []
    predictions, all_token_preds = [], []
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

        token_preds = [(token, NER_MAP[token.ent_type_]) for token in sent_res[start_index:new_end_index]] #todo possibly may filter out if not all ent types are the same / not all in the same span of ent
        all_token_preds.append(token_preds)
        if len(token_preds) == 1:
            predictions.append(token_preds[0][1])
    return predictions, all_token_preds


        # add data to results file
#TODO maybe can change the condition here. can check if we have predicted an overall correct tag for the token range. ex. part of a location, ex. part of a name. need to give some thoght
def predict_on_spacy_mistakes(data: list[conllu.models.TokenList]):
    correct_preds, total_tokens = 0, 0
    k_s = [1]
    res_dict = {str(k): {"correct_preds": 0} for k in k_s}
    # masked_ners = {}
    for index, sent in enumerate(data):
        print(index)
        # predict named entities in sentence
        sent_text = " ".join([t['form'] for t in sent])
        sent_nlp = nlp(sent_text)
        # see which ners spacy did not identify correctly, and mask them and retag
        for token, conllu_token in zip(sent_nlp, sent):
            token_ent_type = NER_MAP[token.ent_type_]
            if token_ent_type == (conllu_token["lemma"][2:] if \
            conllu_token["lemma"].startswith("I-") or conllu_token["lemma"].startswith("B-") else conllu_token["lemma"]):
                for k in k_s:
                        preds, token_preds = get_named_entity_after_masking(sent, start_index=token.i, end_index=token.i+1, k=int(k))
                        if not preds:
                            continue
                        max_pred = max(set(preds), key=preds.count)
                        if max_pred == (conllu_token["lemma"][2:] if \
                    conllu_token["lemma"].startswith("I-") or conllu_token["lemma"].startswith("B-") else conllu_token["lemma"]):
                            res_dict[str(k)]["correct_preds"] += 1
                total_tokens += 1
    print(res_dict)
    print(total_tokens)
    # print(correct_preds/total_tokens)




if __name__ == '__main__':
    data = load_data(r"/cs/snapless/gabis/shaharspencer/CreativeLanguageProject/src/masking_subproject/NER/raw_data/en_ewt-ud-test.conllu")
    predict_on_spacy_mistakes(data)