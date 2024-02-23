import sys
import os
sys.path.append('/cs/snapless/gabis/shaharspencer/CreativeLanguageProject/src/')
sys.path.append('C:\\Users\\User\\PycharmProjects\\CreativeLanguage')
sys.path.append('C:\\Users\\User\\PycharmProjects\\CreativeLanguage\\src')

# h
sys.path.append(r'/cs/snapless/gabis/shaharspencer')
sys.path.append(r'/cs/snapless/gabis/shaharspencer/CreativeLanguageProject')
sys.path.append(r'/cs/snapless/gabis/shaharspencer/CreativeLanguageProject/')
parent_dir = os.path.abspath(r'CreativeLanguageProject/src')

# Append the parent directory to sys.path

sys.path.append(parent_dir)

print(sys.path)

import conllu.models


from base_functions import get_spacy_ners_from_conllu_sent, get_nlp, get_gold_ner, load_data, get_spacy_ners_from_list_sent, NER_MAP
from src.masking_subproject.tagging.tag_with_mask import FillMask

# fill_masks = {"1": FillMask(top_k=1), "2": FillMask(top_k=2), "3": FillMask(top_k=3)}
fill_masks = {"1": FillMask(top_k=1),
              "2": FillMask(top_k=2)}
def replace_tokens(tokens:list[str], start_index:int, end_index: int, replecament_token: str):
    masked_tokens = tokens[:start_index] + [replecament_token] + tokens[end_index:]
    return masked_tokens

nlp = get_nlp()

def get_named_entity_after_masking(sent: conllu.models.TokenList, start_index:int, end_index:int, k:int)->list[str]:
    res = []
    predictions, all_token_preds = [], []
    data_tuples = []
    fill_mask = fill_masks[str(k)]
    sent_text_list = [str(w) for w in sent]
    masked_sentence = replace_tokens(sent_text_list, start_index=start_index, end_index=end_index, replecament_token="<mask>")
    sent_text = " ".join(masked_sentence)
    fillmask_res = fill_mask.replace_token(sent_text)
    for r in fillmask_res:
        resulted_sentence = replace_tokens(tokens=sent_text_list, start_index=start_index, end_index=end_index, replecament_token=r["token_str"])
        original_span_length = end_index - start_index
        replacement_length = len(r["token_str"].split())
        if replacement_length == 1:
            data_tuples.append((" ".join(resulted_sentence), {}))
        # joint_sent, flag = " ".join(resulted_sentence), False
        # sent_res = nlp(joint_sent)
        # replacement_length = len(r["token_str"].split())
        # len_diff = replacement_length - original_span_length
        # new_end_index = end_index + len_diff
        #
        # token_preds = [(token, NER_MAP[token.ent_type_]) for token in sent_res[start_index:new_end_index]] #todo possibly may filter out if not all ent types are the same / not all in the same span of ent
        # all_token_preds.append(token_preds)
        # if len(token_preds) == 1:
        #     predictions.append(token_preds[0][1])

    # return predictions, all_token_preds
    return data_tuples


        # add data to results file
#TODO maybe can change the condition here. can check if we have predicted an overall correct tag for the token range. ex. part of a location, ex. part of a name. need to give some thoght
def predict_on_spacy_mistakes(data: list[conllu.models.TokenList]):
    correct_preds, total_tokens = 0, 0
    k_s = [1, 2]
    res_dict = {str(k): {"correct_preds": 0} for k in k_s}
    data_tuples = [(" ".join([t['form'] for t in sent]), {"conllu_sent": sent})  for sent in data]
    z = 0
    #
    # masked_ners = {}
    for doc, context in nlp.pipe(data_tuples, batch_size=1000,
                                      as_tuples=True,
                                      n_process=1):
    # for index, sent in enumerate(data):

        print(z, flush=True)
        z += 1
        # predict named entities in sentence
        # sent_text = " ".join([t['form'] for t in sent])
        # sent_nlp = nlp(sent_text)
        # see which ners spacy did not identify correctly, and mask them and retag
        # for token, conllu_token in zip(sent_nlp, sent):
        #     process_list = []
        for token in doc:
            conllu_token = context["conllu_sent"][token.i]

            token_ent_type = NER_MAP[token.ent_type_]
            if token_ent_type == (conllu_token["lemma"][2:] if conllu_token["lemma"].startswith("I-") or conllu_token["lemma"].startswith("B-") else conllu_token[
                "lemma"]):
                for k in k_s:
                    # preds, token_preds = get_named_entity_after_masking(context["conllu_sent"], start_index=token.i,
                    #                                                     end_index=token.i + 1, k=int(k))
                    token_k_data_tuples = get_named_entity_after_masking(context["conllu_sent"], start_index=token.i,
                                                                 end_index=token.i + 1, k=int(k))
                    preds = []
                    for processed_sentence, c in nlp.pipe(token_k_data_tuples, batch_size=5,
                                      as_tuples=True,
                                      n_process=1):
                        preds.append(NER_MAP[processed_sentence[token.i].ent_type_])
                    if not preds:
                        continue
                    max_pred = max(set(preds), key=preds.count)
                    if max_pred == (conllu_token["lemma"][2:] if conllu_token["lemma"].startswith("I-") or conllu_token["lemma"].startswith("B-") else conllu_token["lemma"]):
                        res_dict[str(k)]["correct_preds"] += 1
                total_tokens += 1
    print(res_dict)
    print(total_tokens)
    print(res_dict)
    print(total_tokens)
    # print(correct_preds/total_tokens)




if __name__ == '__main__':
    data = load_data(r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\NER\raw_data\en_ewt-ud-test.conllu")
    # data = load_data(r"/cs/snapless/gabis/shaharspencer/CreativeLanguageProject/src/masking_subproject/NER/raw_data/en_ewt-ud-test.conllu")
    predict_on_spacy_mistakes(data)