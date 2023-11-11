"""
This file converts the raw conllu data to the following sentence format:
John loves Mary
Bobby likes to run
"""

from docopt import docopt

usage = '''
word_pos_converter CLI.
number of 
Usage:
    word_pos_converter.py <file_to_proccess> <n_sentences>
'''

from conllu import parse

def convert_conllu_to_raw_sentences(conllu_content, sentence_limit=10):
    raw_sentences = []
    current_sentence = []
    sentence_count = 0

    for sentence in conllu_content:
        for token_info in sentence:
            current_sentence.append(token_info['form'])
        raw_sentences.append(" ".join(current_sentence))
        current_sentence = []
        sentence_count += 1

        if sentence_limit is not None and sentence_count >= sentence_limit:
            break

    return raw_sentences





if __name__ == '__main__':
    args = docopt(usage)
    file_to_process = args["<file_to_proccess>"]
    n_sentences = int(args["<n_sentences>"]) if args["<n_sentences>"] != \
                                                "None" else None


    with open(file_to_process, 'r', encoding='utf-8') as conllu_file:
        conllu_content = parse(conllu_file.read())

    raw_sentences = convert_conllu_to_raw_sentences(conllu_content, n_sentences)

    output_file = f'../files/output_raw_sentences_{n_sentences}_sentences.txt'

    with open(output_file, 'w', encoding='utf-8') as output:
        for sentence in raw_sentences:
            output.write(sentence + '\n')

    print(f'data converted and saved to {output_file} with raw sentences')

