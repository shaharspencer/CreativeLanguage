import logging
import time

import pandas as pd
import spacy
import pandas
from spacy.tokens import DocBin
from docopt import docopt
from pathlib import Path
from spacy.symbols import ORTH
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
import os
from tqdm import tqdm

usage = '''
Processor CLI.
Usage:
    processor.py [<file_to_process>] [<number_of_blogposts>]
'''

"""
Creates a .spacy doc_bin of given csv file.

    Paremeters:
        file(string): file from which to create .spacy file
        number_of_blogposts(int): limit on how many blogposts to process

    Returns: 
        None
"""

class Processor:
    def __init__(self,
                 source_file =
                 r"C:\Users\User\OneDrive\Documents\CreativeLanguageOutputFiles\training_data\blogtext_files\blogtext.csv",
                 output_file_directory = r"C:\Users\User\OneDrive\Documents\CreativeLanguageOutputFiles\training_data\spacy_data\withought_context_lg_model",

                 model = "en_core_web_lg", number_of_blogposts=50000,
                 output_file_name="data_from_first_50000_lg_model.spacy"
                 ):
        # the name of the file we want to write to
        self.output_file_directory = output_file_directory
        self.number_of_blogposts = number_of_blogposts
        self.output_file_name = output_file_name
        # a .csv file that contains the unprocessed data
        self.source_file = source_file
        # create a dataframe from the .csv file
        self.df = pandas.read_csv(self.source_file, encoding = 'utf-8'
                                  ).head(self.number_of_blogposts)
        # initialize a docBin object with the following attributes
        self.doc_bin = DocBin(
            attrs=["ORTH", "TAG", "HEAD", "DEP", "ENT_IOB", "ENT_TYPE",
                   "ENT_KB_ID", "LEMMA", "MORPH", "POS"],
            store_user_data=True)
        # initialize a spacy nlp object based on "model" type
        self.nlp = spacy.load(model)
        # add attributes to nlp object: ex. don't split hyphens
        self.add_attrs_to_nlp()

    def add_attrs_to_nlp(self, no_split_hyphens = False,
                         add_i_m_case = True):
        if add_i_m_case:
            special_case = [{ORTH: "i"}, {ORTH: "'m"}]
            self.nlp.tokenizer.add_special_case("i'm", special_case)
            special_case = [{ORTH: "I"}, {ORTH: "'m"}]
            self.nlp.tokenizer.add_special_case("I'm", special_case)

        if no_split_hyphens:
            self.nlp.tokenizer.infix_finditer = self.recompile_hyphens


    """
        iterate over rows in dataframe and create nlp object from text.
        can limit number of bloposts via number_of_blogposts argument
        push docBin to disk (save docBin)
    """
    def process_file_and_create_nlp_objs(self):
        output_path = os.path.join(self.output_file_directory,
                                   self.output_file_name)

        with tqdm(total=self.df.shape[0]) as pbar:
            for index, row in self.df.iterrows():
                pbar.update(1)
                self.add_blogpost_to_docbin(index, row)

        self.doc_bin.to_disk(output_path)



    """
        adds each sentence in a single blogpost to the docBin. 
        first creates a nlp object from the entire blogpost to get seperate
        sentences.
        then for each sentence create an nlp object, add the doc index etc
        as user data, and save to docbin 
    """
    def add_blogpost_to_docbin(self, index, row: pd.DataFrame):
        blogpost_sents = self.nlp(row['text']).sents

        for sent_index, sent in enumerate(blogpost_sents):
            sentence = sent.text.strip()
            doc = self.nlp(sentence) # proccesed sentence
            doc.user_data = {"DOC_INDEX": index,
                             "SENT_INDEX": sent_index}
            for col_name, col_val in row.iteritems():
                doc.user_data[col_name] = col_val
            doc.retokenize()
            self.doc_bin.add(doc)



    def recompile_hyphens(self):
        infixes = (
                          LIST_ELLIPSES
                          + LIST_ICONS
                          + [
                              r"(?<=[0-9])[+\-\*^](?=[0-9-])",
                              r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                                  al=ALPHA_LOWER, au=ALPHA_UPPER, q=
                                  CONCAT_QUOTES
                              ),
                              r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),

                      r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        ]
        )

        infix_re = compile_infix_regex(infixes)
        return infix_re.finditer


if __name__ == '__main__':
    args = docopt(usage)
    if args['<file_to_process>']:
        source_file = args['<file_to_process>']

        proccessor = Processor(source_file=source_file)
    else:
        processor = Processor()

    if args['<number_of_blogposts>']:
        number_of_files = args['<number_of_blogposts>']
        proccessor.number_of_blogposts = number_of_files

    processor.process_file_and_create_nlp_objs()

