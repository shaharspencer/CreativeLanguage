import collections
import concurrent.futures
import sys
import multiprocessing

# multiprocessing.set_start_method('spawn', True)


# TODO utilize GPU?
# TODO save absolute index OR in verb_path, save index of verb AND indexes in sentence for replacement
# TODO save tokenized list??? for prections of pos
# TODO check if the pos are correct

# Add missing paths
# sys.path.append('C:\\Users\\User\\PycharmProjects\\CreativeLanguageWithVenv')
# sys.path.append('C:\\Program Files\\JetBrains\\PyCharm 2022.2.1\\plugins\\python\\helpers\\pycharm_display')
# sys.path.append('C:\\Program Files\\JetBrains\\PyCharm 2022.2.1\\plugins\\python\\helpers\\pycharm_matplotlib_backend')

sys.path.append('/cs/snapless/gabis/shaharspencer/CreativeLanguageProject/src')
# h
sys.path.append(r'/cs/snapless/gabis/shaharspencer')

import pandas as pd
import spacy


activated = spacy.prefer_gpu()
print(f"activated gpu: {activated}\n")
print(f"spacy version: {spacy.__version__}\n")
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

parent_dir = os.path.abspath(r'CreativeLanguageProject/src')

# Append the parent directory to sys.path


sys.path.append(
    r"C:\Users\User\PycharmProjects\CreativeLanguageWithVenv\src\generate_and_test_spacy\processors\ensemble_tagger.py")

sys.path.append(parent_dir)
print(sys.path)
from src.generate_and_test_spacy.processors import ensemble_tagger

# from src.utils.path_configurations import files_directory, \
#     training_data_files_directory, spacy_files_directory


usage = '''
processor CLI.
Usage:
    processor.py <file_to_process> <number_of_blogposts> <to_conllu>
'''

tagger = ensemble_tagger.EnsembleTagger()

"""
  use this method to apply custom pos tagger
  @:param doc tokenized doc object 
  @:return doc object with changed pos_ components
"""


@Language.component("custom_tagger")
def multi_tagger(doc):
    if not doc:
        return doc
    tags = tagger.get_tags_list(doc)
    sorted_values = [tags[key] for key in sorted(tags.keys())]

    for token, (text, tag) in zip(doc, sorted_values):
        if tag != "X":
            token.pos_ = tag
    return doc


"""
Creates a .spacy doc_bin of given csv file.

    Paremeters:
        source_file(string): csv file from which to create .spacy file
        model(str) : model for spaCy to utilize, defaults to en_core_web_lg
        number_of_blogposts(int): limit on how many blogposts to process
"""


class Processor:
    def __init__(self, to_conllu, use_ensemble_tagger,
                 source_file=
                 r"blogtext.csv",
                 model="en_core_web_lg", number_of_blogposts=40000,
                 to_process=True
                 ):
        self.to_conllu = to_conllu
        self.__load_nlp_objects(model, use_ensemble_tagger)
        self.__add_attrs_to_nlp()
        self.__load_docbin()
        if to_process:
            self.__load_attributes(source_file, number_of_blogposts)

    """
        loads nlp object with requested model
        adds extra pipelines if relevant
        @:param model(str): model to load for spaCyddddd
    """

    def __load_nlp_objects(self, model, use_ensemble_tagger):
        self.basic_nlp = spacy.load(model)
        self.ensemble_nlp = None
        # add custom tagger to end of pipeline
        if use_ensemble_tagger:
            self.ensemble_nlp = spacy.load(model)
            self.ensemble_nlp.add_pipe("custom_tagger", after="ner")
        if self.to_conllu:
            if self.ensemble_nlp:
                self.ensemble_nlp.add_pipe("conll_formatter",
                              last=True)  # remove if serializing data
            else:
                self.basic_nlp.add_pipe("conll_formatter",
                              last=True)

    """
       ddddddd load docBin object to store serialized data
    """

    def __load_docbin(self):
        self.doc_bin = DocBin(
            attrs=["ORTH", "TAG", "HEAD", "DEP", "ENT_IOB", "ENT_TYPE",
                   "ENT_KB_ID", "LEMMA", "MORPH", "POS"],
            store_user_data=True)

    """
        load relevant attributes such as source file paths
        and output file paths
    """

    def __load_attributes(self, source_file, number_of_blogposts):

        # get a .csv file that contains the unprocessed data
        # self.source_file_path = os.path.join(files_directory,
        #                                      training_data_files_directory,
        #                                      source_file)

        self.source_file_path = source_file
        # the name of the file we want to write to
        self.output_file_path = "data_from_first_{n}_lg_model_spacy_3.5.5." \
                                "spacy".format(
            n=number_of_blogposts)
        # self.output_file_path = os.path.join(files_directory,
        #                                      spacy_files_directory,
        #                                      output_file_dir,
        #                                      self.output_file_name)

        self.blogpost_limit = number_of_blogposts
        # create a dataframe from the .csv file
        self.df = pandas.read_csv(self.source_file_path, encoding='utf-8')

    def __add_attrs_to_nlp(self, no_split_hyphens=False):
        if no_split_hyphens:
            self.nlp.tokenizer.infix_finditer = self.recompile_hyphens

    """
        iterate over rows in dataframe and create nlp object from text.
        can limit number of blogposts via number_of_blogposts attribute
        push docBin to disk (save docBin)
    """

    def process_file(self) -> None:
        self.__set_doc_extensions()
        print(f"processing file!\n", flush=True)
        print(f"cpus available: {os.cpu_count()}\n", flush=True)
        with tqdm(total=self.blogpost_limit) as pbar:
            final_docs = {}
            # submit each task
            with concurrent.futures.ProcessPoolExecutor(max_workers=
                                                        3) \
                    as executor:
                futures = []
                # submit each blogpost for processing
                for index in range(self.blogpost_limit):
                    pbar.update(1)
                    row = self.df.loc[index]
                    print(f"submitting blogpost number {index}\n", flush=True)
                    future = executor.submit(self.proccess_blogpost,
                                             index, row)
                    futures.append(future)
                # wait for completion of tasks and add to dictionary
                for task in concurrent.futures.as_completed(futures):
                    index, docs = task.result()
                    final_docs[index] = docs
                    print(
                        f"completed processing blogpost number {index}, "
                        f"total done are {len(final_docs)}\n",
                        flush=True)

            # sort dict items according to blogpost number
            final_docs = collections.OrderedDict(sorted(final_docs.items()))
            print(f"length of docs gathered is {len(final_docs)}\n",
                  flush=True)
            assert len(final_docs) == self.blogpost_limit
            # if we want to save conllu format
            if self.to_conllu:
                self.save_conllu_format(final_docs)
            # if we want to save to docBin
            else:
                # finally add each blogposts sentences
                for doc in final_docs.keys():
                    for sent_doc in final_docs[
                        doc]:  # for each sent in the blogpost
                        self.doc_bin.add(sent_doc)

                # save to disk
                self.doc_bin.to_disk(self.output_file_path)

    """
        sets Doc extensions to save metadata about each entry in spaCy object
    """

    def __set_doc_extensions(self):
        Doc.set_extension("DOC_INDEX", default=None)
        Doc.set_extension("SENT_INDEX", default=None)
        Doc.set_extension("ORIGINAL_SENTENCE", default=None)

    """
        processes a sentence into a Doc object.
        @:param sentence(str): string to process
        @:return doc(Doc): spacy doc object
    """

    def process_text(self, sentence: str) -> Doc:
        doc = self.ensemble_nlp(sentence) if self.ensemble_nlp \
            else self.basic_nlp(sentence)
        doc.retokenize()
        return doc

    """
        adds each sentence in a single blogpost to the docBin.
        first creates a nlp object from the entire blogpost to get seperate
        sentences.
        then for each sentence create an nlp object, add the doc index etc
        as user data, and save to docbin
        @:param index(int): index of row to save to doc
        @ row(pd.dataFrame): row #index from main dataframe
        @:return docs(list[int, [Doc]]) list with first entry being the 
        blogpost number and the second a list of the generated Docs
    """

    def proccess_blogpost(self, index, row: pd.DataFrame) -> list[Doc]:
        docs = []
        blogpost_text = self.__clean_text_data(row['text'])
        blogpost_sents = self.basic_nlp(blogpost_text).sents

        for sent_index, sent in enumerate(blogpost_sents):
            original_sentence = sent.text
            sentence = self.__normalize_sent(sent.text)
            doc = self.process_text(sentence)
            doc.user_data["DOC_INDEX"] = index
            doc.user_data["SENT_INDEX"] = sent_index
            doc.user_data["ORIGINAL_SENTENCE"] = original_sentence
            for col_name, col_val in row.items():
                doc.user_data[col_name] = col_val
            doc.retokenize()
            docs.append(doc)
            # self.doc_bin.add(doc)
            print(f"doc index: {index}, sent index: {sent_index}\n",
                  flush=True)
        return [index, docs]

    """
        recompile hyphens for tokenizer so that words with hyphens between
        them are not split while tokenizing text
    """

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
        @:param blogpost(str): blogpost text
        @:return blogpost(str): cleaned blogpost text
    """

    def __clean_text_data(self, blogpost: str) -> str:
        # clean html expressions
        blogpost = blogpost.replace("&nbsp;", " ")
        blogpost = blogpost.replace("nbsp;", " ")
        blogpost = blogpost.replace("&nbsp", " ")
        blogpost = blogpost.replace("&amp;", "&")
        blogpost = blogpost.replace("amp;", "&")
        blogpost = blogpost.replace("&amp", "&")
        blogpost = blogpost.strip()

        return blogpost

    """
        normalize sentence data for spacy pipeline
        @:param sentence(str): sentence to process
        @:return sentence(str): normalized sentence
    """

    def __normalize_sent(self, sentence: str) -> str:
        sentence = sentence[0] + sentence[1:].lower()
        sentence = sentence.strip()
        return sentence

    """
        save docs conllu attributes 
        @param final_docs(dict[list[Doc]): dictionary of (blogpost number, generated docs)
                                            to save conllu formats of
    """

    def save_conllu_format(self, final_docs: dict[list[Doc]]):
        for doc in final_docs.keys():
            for sent_doc in final_docs[
                doc]:  # for each sentence in the blogpost
                pass


if __name__ == '__main__':
    args = docopt(usage)

    source_file = args['<file_to_process>']

    number_of_files = int(args['<number_of_blogposts>'])

    to_conllu = True if args["<to_conllu>"] == "True" else False

    processor = Processor(source_file=source_file,
                          number_of_blogposts=number_of_files,
                          to_conllu=to_conllu,
                          use_ensemble_tagger=True)

    processor.process_file()
