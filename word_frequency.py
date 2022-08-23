import spacy
from spacy.tokens import DocBin
from collections import Counter
import csv
import pandas as pd
from docopt import docopt


usage = '''
open_processed_file CLI.
Usage:
    word_frequency.py <create_csv> <spacy_path> <word_freq_csv> <create_hist> <k> <least_or_most> <hist_path>
'''


# load model
nlp = spacy.load('en_core_web_trf')



"""
Creates a word frequency csv given a .spacy file

    Paremeters:
        spacy_path(string): a path to a .spacy file from which to create the csv
        csv_path: file to open and write frequency of words to
    Returns: 
        None
"""
def word_frequency_csv(spacy_path, csv_path):
    doc_bin = DocBin().from_disk(spacy_path)
    words = []
    for doc in list(doc_bin.get_docs(nlp.vocab)):
        words += [token.text for token in doc if not
        (token.pos_ == "PUNCT" or token.is_space)]

    with open(csv_path, 'w', encoding='utf-8') as f:
        fieldnames = ["word", "frequency"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        word_counter = Counter(words)
        word_counter = sorted(word_counter.items(),
                              key=lambda pair: pair[1], reverse=True)
        writer.writerow({'word': "word", 'frequency': "frequency"})
        for word in word_counter:
            writer.writerow({'word': word[0],'frequency': word[1]})

"""
Creates a histogram of k most or least frequent words, 
given csv file of words and their frequency.

    Paremeters:
        csv_path (string): path to csv file with two colnames, "word" and "frequency"
        k(int): number of words in histogram
        side(str): either "most" or "least" (frequent)
    Returns: 
        None
"""
def create_word_histogram(csv_path:str, k:int, side:str, hist_path:str):
    data = pd.read_csv(csv_path)
    df = pd.DataFrame(data)
    if side == "most":

        x_axis = df[df.columns[0]][:k]
        y_axis = df[df.columns[1]][:k]
        from matplotlib import pyplot as plt
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        plt.bar(x_axis, y_axis)
        plt.title("Word frequency histogram in blog text corpus, "
                  ""+str(k)+" most frequent words")
        plt.xlabel("Word")
        plt.ylabel("Frequency")
        plt.xticks(rotation=90, size=4.5)
        plt.savefig(hist_path)
    # if side == "least":
        # x_axis = df['word'][k]
        # y_axis = df['frequency'][:k]
        # from matplotlib import pyplot as plt
        # plt.bar(x_axis, y_axis)
        # plt.title("Word frequency histogram in blog text corpus, "
        #           ""+k+" most frequent words")
        # plt.xlabel("Word")
        # plt.ylabel("Frequency")
        # plt.xticks(rotation=90, size=5)


if __name__ == '__main__':
    args = docopt(usage)
    if args["<create_csv>"] == "True":
        word_frequency_csv(args["<spacy_path>"], args["<word_freq_csv>"])
    if args["<create_hist>"] == "True":
        create_word_histogram(args["<word_freq_csv>"], int(args["<k>"]),
                              args["<least_or_most>"], args["<hist_path>"])