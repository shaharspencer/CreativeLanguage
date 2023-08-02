import collections
import sys



# Add missing paths
# sys.path.append('C:\\Users\\User\\PycharmProjects\\CreativeLanguageWithVenv')
# sys.path.append('C:\\Program Files\\JetBrains\\PyCharm 2022.2.1\\plugins\\python\\helpers\\pycharm_display')
# sys.path.append('C:\\Program Files\\JetBrains\\PyCharm 2022.2.1\\plugins\\python\\helpers\\pycharm_matplotlib_backend')

sys.path.append('/cs/snapless/gabis/shaharspencer/CreativeLanguageProject/src')
# h
sys.path.append(r'/cs/snapless/gabis/shaharspencer')


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
number of 
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

    for token, (text, tag) in zip(doc, tags):
        assert token.text == text
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

"""
    cleans out blogpost from &nsbp; nbsp, &amp;, amp, &nbsp, nbsp;, etc
    these are all html remanants
    @:param blogpost(str): blogpost text
    @:return blogpost(str): cleaned blogpost text
"""
def __clean_text_data(blogpost: str) -> str:
    # clean html expressions
    blogpost = blogpost.replace("&nbsp;", " ")
    blogpost = blogpost.replace("nbsp;", " ")
    blogpost = blogpost.replace("&nbsp", " ")
    blogpost = blogpost.replace("&amp;", "&")
    blogpost = blogpost.replace("amp;", "&")
    blogpost = blogpost.replace("&amp", "&")
    blogpost = blogpost.strip()
    return blogpost


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
        @:param model(str): model to load for spaCy
    """

    def __load_nlp_objects(self, model, use_ensemble_tagger):
        if not use_ensemble_tagger:
            self.nlp = spacy.load(model)
        else:
            self.nlp = spacy.load(model)
            self.nlp.add_pipe("custom_tagger", after="ner")

        if self.to_conllu:
            self.nlp.add_pipe("conll_formatter",
                          last=True)

    """
       load docBin object to store serialized data
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
        self.output_file_path = "GPU_7_30_2023_data_from_first_{n}_lg_model_spacy_3.5.5.spacy".format(n=number_of_blogposts)
        # self.output_file_path = os.path.join(files_directory,
        #                                      spacy_files_directory,
        #                                      output_file_dir,
        #                                      self.output_file_name)

        self.blogpost_limit = number_of_blogposts
        # create a dataframe from the .csv file
        self.df = pandas.read_csv(self.source_file_path, encoding='utf-8')
        self.df['text'] = self.df['text'].fillna('')

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
        data = self.df.head(self.blogpost_limit) if self.blogpost_limit\
            else self.df
        data_tuples = [(self.normalize_text(row["text"]),
                        {col: row[col] for col in self.df.columns})
                       for row in data.to_dict(orient="records")]
        for doc, context in self.nlp.pipe(data_tuples, batch_size=500,
                                          as_tuples=True,
                                          n_process=10):
            # add user data to doc
            for col_name, col_val in context.items():
                doc.user_data[col_name] = col_val
            self.doc_bin.add(doc)
            doc_index = context["doc_index"]
            sent_index = context["sent_index"]
            print(f"done processing doc # {doc_index}, sent # {sent_index}",
                  end=" ",
                  flush=True)
        self.doc_bin.to_disk(self.output_file_path)

    def normalize_text(self, text: str) -> str:
        """
            normalize sentence data for spacy pipeline
            @:param doc(str): doc to process
            @:return sentence(str): normalized sentence
        """
        if not text:
            return text
        text = text[0] + text[1:].lower()
        text = text.strip()
        return text

    """
        sets Doc extensions to save metadata about each entry in spaCy object
    """

    def __set_doc_extensions(self):
        Doc.set_extension("doc_index", default=None)
        Doc.set_extension("sent_index", default=None)
        Doc.set_extension("sent", default=None)

    """
        processes a sentence into a Doc object.
        @:param sentence(str): string to process
        @:return doc(Doc): spacy doc object
    """

    def process_text(self, sentence: str) -> Doc:
        doc = self.nlp(sentence)
        return doc


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


if __name__ == '__main__':
    args = docopt(usage)

    source_file = args['<file_to_process>']

    number_of_files = args['<number_of_blogposts>']
    if number_of_files != "None":
        number_of_files = int(args['<number_of_blogposts>'])
    else:
        number_of_files = 0


    to_conllu = True if args["<to_conllu>"] == "True" else False

    processor = Processor(source_file=source_file,
                          number_of_blogposts=number_of_files,
                          to_conllu=to_conllu,
                          use_ensemble_tagger=True)

    processor.process_file()
