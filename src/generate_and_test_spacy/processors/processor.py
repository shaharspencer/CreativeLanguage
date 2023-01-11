import logging
import time

import pandas as pd
import spacy
import pandas
from spacy.tokens import Doc
from spacy.tokens import DocBin
from docopt import docopt
from pathlib import Path
from spacy.symbols import ORTH
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
import os
from tqdm import tqdm

import utils.path_configurations as paths

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
                r"blogtext.csv",
                 output_file_dir= r"withought_context_lg_model",
                 model="en_core_web_lg", number_of_blogposts=40000,

                 ):
        # get a .csv file that contains the unprocessed data
        self.source_file_path = os.path.join(paths.files_directory,
                                             paths.training_data_files_directory,
                                             source_file)
        # the name of the file we want to write to
        output_file_name = "data_from_first_{n}_lg_model.spacy".format(
            n=number_of_blogposts)
        self.output_file_path = os.path.join(paths.files_directory,
                                             paths.spacy_files_directory,
                                             output_file_dir,
                                             output_file_name)

        self.number_of_blogposts = number_of_blogposts

        # create a dataframe from the .csv file
        self.df = pandas.read_csv(self.source_file_path, encoding ='utf-8'
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
                         add_i_m_case=True):
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
        Doc.set_extension("DOC_INDEX", default=None)
        Doc.set_extension("SENT_INDEX", default=None)


        with tqdm(total=self.df.shape[0]) as pbar:
            for index, row in self.df.iterrows():
                pbar.update(1)
                self.add_blogpost_to_docbin(index, row)

        self.doc_bin.to_disk(self.output_file_path)



    """
        adds each sentence in a single blogpost to the docBin. 
        first creates a nlp object from the entire blogpost to get seperate
        sentences.
        then for each sentence create an nlp object, add the doc index etc
        as user data, and save to docbin 
    """
    def add_blogpost_to_docbin(self, index, row: pd.DataFrame):

        blogpost_text = self.clean_text_data(row['text'])
        blogpost_sents = self.nlp(blogpost_text).sents

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

    """
    cleans out blogpost from &nsbp; nbsp, &amp;, amp, &nbsp, nbsp;, etc
    these are all html remanants
    """
    def clean_text_data(self, blogpost:str)->str:
        if "&nbsp;" in blogpost:
            x = 0
        blogpost = blogpost.replace("&nbsp;", " ")
        blogpost = blogpost.replace("nbsp;", " ")
        blogpost = blogpost.replace("&nbsp", " ")
        blogpost = blogpost.replace("&amp;", "&")
        blogpost = blogpost.replace("amp;", "&")
        blogpost = blogpost.replace("&amp", "&")
        blogpost = blogpost.strip()
        return blogpost




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

