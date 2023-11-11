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

from docopt import docopt

usage = '''
word_pos_converter CLI.
number of 
Usage:
    word_pos_converter.py <file_to_proccess> <n_sentences>
'''

from conllu import parse
from typing import List

def convert_conllu_to_custom_format(conllu_content: List[List[dict]],
                                    output_file, sentence_limit):
    with open(output_file, 'w') as f:
        sentence_count = 0
        for sentence in conllu_content:
            for token_info in sentence:
                if token_info["xpos"] == None:
                    continue
                word = token_info['form']
                pos_tag = token_info['upos']
                f.write(f"{word} {pos_tag}\n")
            f.write("\n")
            sentence_count += 1

            if sentence_limit is not None and sentence_count >= sentence_limit:
                break


def run(file_to_process: str, n_sentences: int | None) -> str:
    output_file = f'../files/tags_data/output_with_pos_UD_tags_{n_sentences}_sentences.txt'

    with open(file_to_process, 'r', encoding='utf-8') as conllu_file:
        conllu_content = parse(conllu_file.read())

    convert_conllu_to_custom_format(conllu_content, output_file, sentence_limit=n_sentences)

    print(f'data converted and saved to {output_file} with UD pos')
    return output_file



if __name__ == '__main__':
    args = docopt(usage)
    file_to_process = args["<file_to_proccess>"]
    n_sentences = int(args["<n_sentences>"]) if args["<n_sentences>"] != \
                                                "None" else None
    run(file_to_process=file_to_process, n_sentences=n_sentences)
