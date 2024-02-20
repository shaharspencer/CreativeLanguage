import conllu.models

from src.masking_subproject.NER.base_functions import get_spacy_ners_from_conllu_sent, get_gold_ner, load_data, get_spacy_ners_from_list_sent
from src.masking_subproject.tagging.tag_with_mask import FillMask

fill_masks = {"1": FillMask(top_k=1)}
def replace_tokens(tokens:list[str], start_index:int, end_index: int, replecament_token: str):
    masked_tokens = tokens[:start_index] + [replecament_token] + tokens[end_index:]
    return masked_tokens

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
        token_len_difference = len(resulted_sentence) - len(sent_text_list)
        ners = get_spacy_ners_from_list_sent(sent=resulted_sentence, sent_id=sent.metadata["sent_id"])
        result_dict = [{"generated_token": r["token_str"],
                        "result_sentence_len_minus_orig_sent_len":
            token_len_difference, "ner_results": ners}]
        res.append(result_dict)

    return res


        # add data to results file

def predict_on_spacy_mistakes(data):
    #TODO need to speed up
    masked_ners = {}
    for sent in data:
        spacy_ners = get_spacy_ners_from_conllu_sent(sent)
        gold_ners = get_gold_ner(sent)
        # see which ners spacy did not identify correctly, and mask them and retag
        spacy_entities = set(
            (ent['text'], ent['label'], ent["start_index"], ent["end_index"]) for ent in spacy_ners)
        gold_entities = set(
            (ent['text'], ent['label'], ent["start_index"], ent["end_index"]) for ent in gold_ners)
        dif = spacy_entities ^ gold_entities

        for item in dif:
            ners = get_named_entity_after_masking(sent, start_index=item[2], end_index=item[3], k=1)
        masked_ners[sent.metadata["sent_id"]] = ners

        z = 0


if __name__ == '__main__':
    data = load_data(r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\NER\raw_data\en_ewt-ud-test.conllu")
    predict_on_spacy_mistakes(data)