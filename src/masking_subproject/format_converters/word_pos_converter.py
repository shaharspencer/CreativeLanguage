"""
This file converts the conllu formatted data to the following format:
John NOUN
loves VERB
Mary NOUN

Bobby NOUN
likes VERB
to PRP
run VERB

with UD pos taggings
"""
import pandas as pd
from docopt import docopt

usage = '''
word_pos_converter CLI.
number of 
Usage:
    word_pos_converter.py <file_to_proccess> <n_sentences>
'''

from conllu import parse
from typing import List

# def convert_conllu_to_custom_format(conllu_content: List[List[dict]],
#                                     output_file, sentence_limit):
#     with open(output_file, 'w', encoding='utf-8') as f:
#         sentence_count = 0
#         for sentence in conllu_content:
#             for token_info in sentence:
#                 if token_info["xpos"] == None or not token_info["form"]:
#                     continue
#                 word = token_info['form']
#                 pos_tag = token_info['upos']
#                 try:
#                     token_id = token_info['id']-1
#                 except Exception: # if id is not saved properly
#                     token_id = token_info['id'][0]-1
#                 try:
#                     s = f"{word} {pos_tag} {sentence_count} {token_id}\n"
#                     f.write(s)
#                 except Exception:
#                     x = 0
#             f.write("\n")
#             sentence_count += 1
#
#             if sentence_limit is not None and sentence_count >= sentence_limit:
#                 break
def convert_conllu_to_dataframe(conllu_content, sentence_limit):
    data = []
    sentence_count = 0
    for sentence in conllu_content:
        for token_info in sentence:
            if token_info["xpos"] is None or not token_info["form"]:
                continue
            word = token_info['form']
            pos_tag = token_info['upos']
            try:
                token_id = token_info['id'] - 1
            except Exception:
                token_id = token_info['id'][0] - 1
            data.append({'Word': word, 'POS_Tag': pos_tag,
                         'Sentence_Count': sentence_count,
                         'Token_ID': token_id})
        sentence_count += 1
        if sentence_limit is not None and sentence_count >= sentence_limit:
            break
    df = pd.DataFrame(data)
    return df


def run(raw_data_file: str, n_sentences: int | None) -> str:
    output_file = f'../files/tags_data/output_with_pos_UD_tags_{n_sentences}_sentences.csv'

    with open(raw_data_file, 'r', encoding='utf-8') as conllu_file:
        conllu_content = parse(conllu_file.read())
    #TODO return functionality of n_sentences
    d = convert_conllu_to_dataframe(conllu_content, sentence_limit=n_sentences)
    d.to_csv(output_file, encoding='utf-8', index=False,sep=',')

    print(f'data converted and saved to {output_file} with UD pos')
    return output_file



if __name__ == '__main__':
    args = docopt(usage)
    file_to_process = args["<file_to_proccess>"]
    n_sentences = int(args["<n_sentences>"]) if args["<n_sentences>"] != \
                                                "None" else None
    # run(file_to_process=file_to_process, se)
