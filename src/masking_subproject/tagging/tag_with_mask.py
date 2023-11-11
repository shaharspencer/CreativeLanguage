import pandas as pd
import spacy
from conllu import parse
from docopt import docopt
from transformers import pipeline
from collections import Counter

usage = '''
tag_with_mask CLI.
number of 
Usage:
    tag_with_mask.py
    tag_with_mask.py <file_to_process> <n_sentences>
'''

from spacy.tokens import Doc


class FillMask:
    def __init__(self, top_k):
        self.classifier = pipeline("fill-mask", "xlm-roberta-large",
                                   top_k=top_k)


    def predict(self, tokenized_text: list[str], index) -> str:
        masked = self.mask_text(sentence_tokens=tokenized_text, index=index)
        replacements = self.replace_token(masked_text=masked)
        r_pos = []
        for r in replacements:
            tagger = nlp(r)
            if not len(tokenized_text) == len(tagger):
                y = 0
            new_pos = tagger[index].pos_
            r_pos.append(new_pos)
        pos_counts = Counter(r_pos)

        # Get the most common POS tag
        most_common_pos = pos_counts.most_common(1)[0][0]

        return most_common_pos

    def mask_text(self, sentence_tokens: list[str], index: int)-> str:
        masked_text = " ".join(["<mask>" if j == index else t for j, t in
                                enumerate(sentence_tokens)])
        return masked_text
    def replace_token(self, masked_text)->list[str]:
        res = []
        replacements = self.classifier(masked_text)
        for r in replacements:
            res.append(masked_text.replace("<mask>", r["token_str"]))

        return res


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load("en_core_web_lg")
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


def mask_and_predict(sentence_text: str,
                     fillmask: FillMask, relevant_indices: list)-> list:
    mask_tags = []
    sentence_splitted = sentence_text.split()

    for i, token in enumerate(sentence_splitted):
        if i in relevant_indices:
            replacement_pos = fillmask.predict(tokenized_text=sentence_splitted, index=i)
            mask_tags.append(replacement_pos)
        else:
            mask_tags.append(" ")
    assert len(mask_tags) == len(sentence_splitted)
    return mask_tags


def convert_conllu_to_masked_tagged_text(conllu_content,
                                         sentence_limit: int | None, combined_df: pd.DataFrame):
    obj = FillMask(top_k=1)
    sentence_count = 0
    preds = []
    for sentence in conllu_content:
        sentence_text = " ".join([str(w) for w in sentence if w["xpos"] != None])
        condition = (combined_df['Sentence_Index'] == sentence_count) & (combined_df['UD_POS'] != combined_df['SPACY_POS'])
        relevant_indices = combined_df.loc[condition, 'Token_Index'].tolist()
        m = mask_and_predict(sentence_text=sentence_text,
                               fillmask=obj, relevant_indices=relevant_indices)
        assert len(m) == len([str(w) for w in sentence if w["xpos"] != None])
        preds = preds + m

        sentence_count += 1

        if sentence_limit is not None and sentence_count >= sentence_limit:
            break
    combined_df["Mask_Tags"] = preds
    return combined_df



def run(raw_data_file: str = r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\files\raw_data\en_ewt-ud-test.conllu",
        n_sentences: int | None = 50, combined_dataframe: str = r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\files\tags_data\UD_Spacy_combined_tags_50_sentences.csv") -> str:
    output_file = f'../files/tags_data/output_with_masked_POS_{n_sentences}_sentences.csv'

    with open(raw_data_file, 'r', encoding='utf-8') as conllu_file:
        conllu_content = parse(conllu_file.read())
    c_df = pd.read_csv(combined_dataframe, sep=' ', header=None,
                       names=['Sentence_Index', 'Token_Index', 'Word', 'UD_POS', 'SPACY_POS'])
    new_combined_df = convert_conllu_to_masked_tagged_text(conllu_content,
                                         sentence_limit=n_sentences, combined_df=c_df)
    new_combined_df.to_csv(output_file, sep=' ', index=False, header=False,
                       encoding='utf-8')
    return output_file




if __name__ == '__main__':
    args = docopt(usage)
    file_to_process = args["<file_to_process>"]
    n_sentences = int(args["<n_sentences>"]) if args["<n_sentences>"] != "None" else None
    if file_to_process == "None" and not n_sentences:
        run()
    else:
        run(raw_data_file=file_to_process, n_sentences=n_sentences)
