"""
This file converts the conllu formatted data to the following format:
John NOUN
loves VERB
Mary NOUN

Bobby NOUN
likes VERB
to PRP
run VERB

with spaCy tags
"""


import spacy
from conllu import parse
from docopt import docopt

usage = '''
word_pos_converter CLI.
number of 
Usage:
    word_pos_converter.py <file_to_proccess> <n_sentences>
'''
from spacy.tokens import Doc

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load("en_core_web_lg")
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


def convert_conllu_to_tagged_text(conllu_content, output_file: str,
                                  sentence_limit: int | None):
    with open(output_file, 'w', encoding='utf-8') as f:
        sentence_count = 0
        for sentence in conllu_content:
            # sentence_text = sentence.metadata["text"]
            sentence_text = " ".join([str(w) for w in sentence if w["xpos"] != None])
            if "Google's" in sentence_text:
                x = 0
            doc = nlp(sentence_text)

            for token in doc:
                f.write(f"{token.text} {token.pos_}\n")

            f.write("\n")
            sentence_count += 1

            if sentence_limit is not None and sentence_count >= sentence_limit:
                break

def run(raw_data_file: str, n_sentences: int | None) -> str:
    output_file = f'../files/tags_data/output_with_pos_SPACY_tags_{n_sentences}_sentences.txt'

    with open(raw_data_file, 'r', encoding='utf-8') as conllu_file:
        conllu_content = parse(conllu_file.read())

    convert_conllu_to_tagged_text(conllu_content, output_file,
                                  sentence_limit=n_sentences)

    print(f'data converted and saved to {output_file} with spaCy POS tags')

    return output_file


if __name__ == '__main__':
    args = docopt(usage)
    file_to_process = args["<file_to_proccess>"]
    n_sentences = int(args["<n_sentences>"]) if args["<n_sentences>"] != \
                                                "None" else None

    run(raw_data_file=file_to_process, n_sentences=n_sentences)

