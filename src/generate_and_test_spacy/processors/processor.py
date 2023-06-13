import sys
print(sys.path)

# Add missing paths
# sys.path.append('C:\\Users\\User\\PycharmProjects\\CreativeLanguageWithVenv')
# sys.path.append('C:\\Program Files\\JetBrains\\PyCharm 2022.2.1\\plugins\\python\\helpers\\pycharm_display')
# sys.path.append('C:\\Program Files\\JetBrains\\PyCharm 2022.2.1\\plugins\\python\\helpers\\pycharm_matplotlib_backend')

# sys.path.append('/cs/snapless/gabis/shaharspencer/CreativeLanguage/src')

sys.path.append('/cs/snapless/gabis/shaharspencer')

import pandas as pd
import spacy
import pandas
from spacy import Language
from spacy.tokens import Doc
from spacy.tokens import DocBin
from docopt import docopt
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
import os
from tqdm import tqdm
sys.path.append('/cs/snapless/gabis/shaharspencer/CreativeLanguage/src/generate_and_test_spacy')
import ensemble_tagger
# from src.utils.path_configurations import files_directory, \
#     training_data_files_directory, spacy_files_directory


usage = '''
Processor CLI.
Usage:
    processor.py <file_to_process> <number_of_blogposts>
'''

"""
Creates a .spacy doc_bin of given csv file.

    Paremeters:
        file(string): file from which to create .spacy file
        number_of_blogposts(int): limit on how many blogposts to process

    Returns: 
        None
"""

"""
  use this method to apply our custom pos tagger
  @:param doc tokenized doc object 
  @:return doc object with changed pos_ components
"""
@Language.component("custom_tagger")
def multi_tagger(doc):
    tags = tagger.get_tags_list(doc)
    sorted_values = [tags[key] for key in sorted(tags.keys())]

    for token, (text, tag) in zip(doc, sorted_values):
        token.pos_ = tag
    return doc

class Processor:
    def __init__(self,
                 source_file =
                r"blogtext.csv",
                 model="en_core_web_lg", number_of_blogposts=40000,
                 ):

        self.load_nlp_objects(model)
        self.add_attrs_to_nlp()
        self.load_docbin()
        self.load_attributes(source_file, number_of_blogposts)

    """
        loads nlp object with requested model
        adds pipelines if relevant
    """
    def load_nlp_objects(self, model):
        self.nlp = spacy.load(model)
        # add custom tagger to end of pipeline
        self.nlp.add_pipe("custom_tagger", last=True)

    """
        load docbin object
    """
    def load_docbin(self):
        self.doc_bin = DocBin(
            attrs=["ORTH", "TAG", "HEAD", "DEP", "ENT_IOB", "ENT_TYPE",
                   "ENT_KB_ID", "LEMMA", "MORPH", "POS"],
            store_user_data=True)
    """
        load relevant attributes such as source file paths
        and output file paths
    """
    def load_attributes(self, source_file, number_of_blogposts):

        # get a .csv file that contains the unprocessed data
        # self.source_file_path = os.path.join(files_directory,
        #                                      training_data_files_directory,
        #                                      source_file)

        self.source_file = source_file
        self.source_file_path = self.source_file
        # the name of the file we want to write to
        self.output_file_path = "data_from_first_{n}_lg_model_spacy_3.5.5." \
                                "spacy".format(
            n=number_of_blogposts)
        # self.output_file_path = os.path.join(files_directory,
        #                                      spacy_files_directory,
        #                                      output_file_dir,
        #                                      self.output_file_name)

        self.number_of_blogposts = number_of_blogposts
        # create a dataframe from the .csv file
        self.df = pandas.read_csv(self.source_file_path, encoding='utf-8')

    def add_attrs_to_nlp(self, no_split_hyphens = False,):
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


        with tqdm(total=self.number_of_blogposts) as pbar:
            for index, row in self.df.iterrows():
                pbar.update(1)
                self.add_blogpost_to_docbin(index, row)
                if index ==self.number_of_blogposts:
                    self.doc_bin.to_disk(self.output_file_path)
                    return

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
            for col_name, col_val in row.items():
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
        @:param blogpost string of the blogpost text
        @:return blogpost text cleaned of nbsp and amp etc
    """
    def clean_text_data(self, blogpost:str)->str:
        blogpost = blogpost.replace("&nbsp;", " ")
        blogpost = blogpost.replace("nbsp;", " ")
        blogpost = blogpost.replace("&nbsp", " ")
        blogpost = blogpost.replace("&amp;", "&")
        blogpost = blogpost.replace("amp;", "&")
        blogpost = blogpost.replace("&amp", "&")
        blogpost = blogpost.strip()
        return blogpost




if __name__ == '__main__':
    tagger = ensemble_tagger.EnsembleTagger()
    args = docopt(usage)

    source_file = args['<file_to_process>']

    number_of_files = int(args['<number_of_blogposts>'])

    proccessor = Processor(source_file=source_file,
                           number_of_blogposts=number_of_files)

    proccessor.process_file_and_create_nlp_objs()

