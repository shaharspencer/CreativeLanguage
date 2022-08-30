import spacy
from spacy.tokens import DocBin
import csv
import pandas as pd
from docopt import docopt


usage = '''
open_processed_file CLI.
Usage:
    word_frequency.py <spacy_path> <verb_freq_csv>
'''

# load model
nlp = spacy.load('en_core_web_trf')


"""
Creates a lemma frequency csv given a .spacy file
Counts frequency of lemmas in different pos

    Paremeters:
        spacy_path(string): a path to a .spacy file from which to create the csv
        csv_path: file to open and write frequency of lemmas in different pos
    Returns:
        None
"""

def lemma_frequency_csv(spacy_path):
    doc_bin = DocBin().from_disk(spacy_path)
    # dict of dicts
    lemmas = {}
    for doc in list(doc_bin.get_docs(nlp.vocab)):
        for token in doc:
            # if token is not a word
            if token.is_punct or token.is_space or token.is_stop or token.pos_ == "PUNCT":
                continue
            # if token lemma is in dict
            if token.lemma_ in lemmas:
                lemmas[token.lemma_][token.pos_] = lemmas \
                [token.lemma_].get(token.pos_, 0) + 1
            # else if not in dict
            else:
                lemmas[token.lemma_] = {token.pos_: 1}

"""
Creates a verb frequency csv given a .spacy file
While only counting the lemmas (walked and walks are treated the same)

    Paremeters:
        spacy_path(string): a path to a .spacy file from which to create the csv
        csv_path: file to open and write frequency of verbs to
    Returns:
        None
"""
def verb_frequency_csv(spacy_path, csv_path):
    doc_bin = DocBin().from_disk(spacy_path)
    verb_dict = {}
    for doc in list(doc_bin.get_docs(nlp.vocab)):
        for token in doc:
            if token.is_punct or token.is_space or token.is_stop:
                continue
            if token.pos_ != "VERB":
                continue
            verb_dict[token.lemma_] = \
                    verb_dict.get(token.lemma_, 0) + 1

    with open(csv_path, 'w', encoding='utf-8') as f:
        fieldnames = ["verb", "frequency"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        word_counter = sorted(verb_dict.items(),
                              key=lambda pair: pair[1], reverse=True)
        writer.writerow({'verb': "verb", 'frequency': "frequency"})
        for word in word_counter:
            writer.writerow({'verb': word[0], 'frequency': word[1]})


if __name__ == '__main__':
    args = docopt(usage)
    lemma_frequency_csv(args['<spacy_path>'])
    verb_frequency_csv(args['<spacy_path>'], args['<verb_freq_csv>'])

