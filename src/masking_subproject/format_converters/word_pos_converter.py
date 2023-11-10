"""
This file converts the conllu formatted data to the following format:
John NOUN
loves VERB
Mary NOUN

Bobby NOUN
likes VERB
to PRP
run VERB
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
                                    output_file, sentence_limit=10):
    with open(output_file, 'w') as f:
        sentence_count = 0
        for sentence in conllu_content:
            for token_info in sentence:
                word = token_info['form']
                pos_tag = token_info['upos']
                f.write(f"{word} {pos_tag}\n")
            f.write("\n")
            sentence_count += 1

            if sentence_limit is not None and sentence_count >= sentence_limit:
                break



if __name__ == '__main__':
    args = docopt(usage)
    file_to_process = args["<file_to_proccess>"]
    n_sentences = int(args["<n_sentences>"])
    output_file = f'../files/output_with_pos_{n_sentences}_sentences.txt'

    with open(file_to_process, 'r', encoding='utf-8') as conllu_file:
        conllu_content = parse(conllu_file.read())

    convert_conllu_to_custom_format(conllu_content, output_file)

    print(f'Data converted and saved to {output_file}')