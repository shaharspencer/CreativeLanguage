import sys
from collections import namedtuple
from enum import Enum

from docopt import docopt
from spacy.tokens import DocBin
import csv
import os
import spacy
print(spacy.__version__)
from spacy.tokens import DocBin
from zipfile import ZipFile
from tqdm import tqdm
sys.path.append(r"CreativeLanguage\src\utils")
sys.path.append(r"CreativeLanguage\src\utils\path_configurations")

print(sys.path)

import src.utils.path_configurations
import src.utils.path_configurations as paths



DictEntry = namedtuple('DictEntry', ['token_text', 'token_sent_text', 'doc_index', 'sent_index',
                                      'token_index', 'tokenized_sentence'])


usage = '''
analyze_verbs CLI.
Usage:
    analyze_verbs.py <spacy_file_name> <num_of_posts>
'''

parts_of_speech = ["VERB", "PROPN", "PART", "NUM", "PUNCT",
                        "ADJ",
                        "ADP", "ADV",
                        "AUX", "CCONJ", "DET", "INTJ", "NOUN", "PRON",
                        "SCONJ", "CONJ",
                        "SYM"]
parts_of_speech_to_ignore = ["X", "SPACE"]

open_class_pos = ["VERB", "PROPN", "NOUN", "ADJ"]

class EXTRA_COLS(Enum):
    PERCENTAGE_AS_OPEN_CLASS_POS = "open class pos / total"
    TOTAL_OPEN_CLASS = "total open class"


#TODO: add % columns


"""
For every word that is used as a verb on some occasion, 
Count the frequency of the word.lemma_ in different pos
And save occurences of that word in the part of speech

    Paremeters:
        spacy_path(string): a path to a .spacy file from which to create the csv
    Returns:
        for_csv(dict): 
         for every word that is a verb, for the lemma of the word,
            for every pos:
                - save sentences in which it is used as that part of speech
                - count occurences
"""


class AnalyzeVerbs:
    def __init__(self, spacy_file_path, num_of_posts, model="en_core_web_lg",
                 ):
        """
            Initializes the AnalyzeVerbs class.

            Args:
                spacy_file_path (str): The path to the spacy file.
                num_of_posts (int): The number of blog posts to analyze.
                model (str, optional): The Spacy model to use. Defaults to "en_core_web_lg".
        """
        # define how many blogposts we want to analyze
        self.number_of_posts = num_of_posts
        # define where the spacy path we want to use is located
        self.spacy_path = self.__define_spacy_path(spacy_file_path=
                                                   spacy_file_path)
        # load file from docBin
        self.doc_bin = DocBin().from_disk(self.spacy_path)

        # define nlp model to use vocab
        self.nlp = spacy.load(model)

        # define which pos we want to collecy
        self.spacy_part_of_speech = parts_of_speech

        # find all words which at some point are classified as verbs
        self.words_classed_as_verb = self.find_all_verbs_in_file()

        #initialize dictionary
        self.verb_dict = {}

        #fill dictionary
        self.analyze_verbs()
    """
        define where the spacy file we want to use is
    """
    def __define_spacy_path(self, spacy_file_path):
        """
              Defines the path to the spacy file.

              Args:
                  spacy_file_path (str): The relative path to the spacy file.

              Returns:
                  str: The full path to the spacy file.
        """
        spacy_path = os.path.join(paths.files_directory,
                                       paths.spacy_files_directory,
                                       spacy_file_path)

        return spacy_path

    #TODO optimize runtime
    """
    creates a set of all verbs that were at some point in the files
    classified as verbs
    """
    def find_all_verbs_in_file(self)->set:
        """
               Retrieves all words classified as verbs from the spacy file.

               Returns:
                   list: A list of tokens classified as verbs.
        """
        verbs = set()
        for doc in self.doc_bin.get_docs(self.nlp.vocab):
            for token in doc:
                if not self.verify_we_want_to_add_token(token):
                    continue
                verbs.add(token.lemma_.lower())
        return verbs

    """
        create dictionary with all words that are at some point classified as verbs
        we know they were classified as verbs using the set words_classed_as_verbs
        save all instances in different parts of speech, lemmas and more info
        @:param doc_limit = doc index to stop at
    """
    def analyze_verbs(self):
        for doc in self.doc_bin.get_docs(self.nlp.vocab):
            doc_index = doc.user_data["doc index"]
            sent_index = doc.user_data["sent index"]
            print(f"processing doc number {doc_index}, "
                  f"sent index {sent_index}\n")
            for token in doc:
                if (token.lemma_.lower() in self.words_classed_as_verb):
                    self.add_token_to_dict(token)

    """
        adds current instance of word to dictionary
        adds template for word if not already in the dictionary
        :return bool True if added
                     False else
    """
    #TODO check if problem was solved for index if i use doc instead of sent
    def add_token_to_dict(self, token)->bool:
        if token.pos_ in parts_of_speech_to_ignore:
            return False
        self.add_token_template_to_dict(token)
        # convert all letters but first letter to lowercase
        dict_entry = DictEntry(token_text=token.text,
                               token_sent_text=token.sent.text,
                               doc_index=token.doc.user_data["doc_index"],
                               sent_index=token.doc.user_data["sent_index"],
                               token_index=token.i,
                               tokenized_sentence=
                               tuple([token.text for token in token.sent]))
        self.verb_dict[token.lemma_.lower()][token.pos_]["Instances"].add \
            (dict_entry)
        self.verb_dict[token.lemma_.lower()][token.pos_]["lemma"] = \
            token.lemma_
        self.verb_dict[token.lemma_.lower()][token.pos_]["Counter"] += 1
        return True

    """
    if token is not in the dictionary add it to the 
    dictionary with a pos template
    """
    def add_token_template_to_dict(self, token:spacy.tokens):
        if not token.lemma_.lower() in self.verb_dict.keys():
            self.verb_dict[token.lemma_.lower()] = self.dict_template()


    """
    create csv with all different count of parts of speech
    """

    def write_dict_to_csv(self, pos_to_use, fields_to_write, additional_cols,
                          output_file_name):
        output_path = self.__define_output_path(output_file_name)
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            fieldnames, pos_dict = self.__define_fieldnames_and_pos_dict(
                additional_cols=additional_cols,
                                                  pos_to_use=pos_to_use,
                                                  fields_to_write=fields_to_write)

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(pos_dict)
            self.writing_to_csv(writer, pos_to_use, fields_to_write, additional_cols)

    def __define_fieldnames_and_pos_dict(self, additional_cols, pos_to_use, fields_to_write):
        fieldnames = ["word"]
        pos_fields, pos_dict = self.csv_pos_template(pos_to_use,
                                                     fields_to_write)
        fieldnames += pos_fields

        additional_cols_as_strs = [field.value for field in additional_cols]
        fieldnames += additional_cols_as_strs

        # add to DictWriter dictionary
        pos_dict.update(self.fieldnames_for_csv(additional_cols_as_strs))
        return fieldnames, pos_dict

    """
        define csv final path according to output_file_name given to object
    """
    def __define_output_path(self, output_file_name):
        output_file_name = "first_{n}_posts_".format(
            n=self.number_of_posts) + output_file_name
        output_path = os.path.join(
            src.utils.path_configurations.files_directory,
            src.utils.path_configurations.morphological_dimension_directory,
            src.utils.path_configurations.morphological_dimension_source_files,
            output_file_name)
        return output_path



    """
        does the actual writing to the csv
        @:param writer: file we are writing to, will be closed in another scope
        @:param pos_to_use: parts of speech we are regarding
        @:param fields_to_write: fields to use in csv, ex: count, %, lemma
        @:param additional cols: extra columns that exist for each verb, for example
        percent in specific types of speech
    """
    def writing_to_csv(self, writer, pos_to_use, fields_to_write, add_cols):
        with tqdm(desc="writing to csv", colour="green",
                  total=len(self.verb_dict.keys())) as pbar:

            for word in self.verb_dict.keys():
                pbar.update(1)
                n_dict = {'word': word}
                total = sum(
                    [self.verb_dict[word][pos]["Counter"] for pos in
                     pos_to_use])
                if EXTRA_COLS.PERCENTAGE_AS_OPEN_CLASS_POS in add_cols:
                    n_dict[EXTRA_COLS.PERCENTAGE_AS_OPEN_CLASS_POS.value] = \
                    total / sum([self.verb_dict[word][pos]["Counter"]
                                          for pos in
                     parts_of_speech])
                if EXTRA_COLS.TOTAL_OPEN_CLASS in add_cols:
                    n_dict[EXTRA_COLS.TOTAL_OPEN_CLASS.value] = \
                        total
                for pos in pos_to_use:
                    # choose fields we want to write to (these should exist
                    # in dictionary)
                    if "count" in fields_to_write:
                        n_dict[pos + "_count"] = self.verb_dict[word][pos][
                        "Counter"]
                    if "%" in fields_to_write:
                        n_dict[pos + "%"] = self.verb_dict[word][pos][
                                            "Counter"] / total
                    if "lemma" in fields_to_write:
                        n_dict[pos + "_lemma"] = self.verb_dict[word][pos]["lemma"]

                writer.writerow(n_dict)



    def create_text_files(self):
        from datetime import datetime
        datetime = datetime.today().strftime('%Y_%m_%d')
        folder_name = r"first_{n}_posts_with_lg".format(n=self.number_of_posts) + datetime
        zip_file_dir = os.path.join(paths.files_directory, paths.morphological_dimension_directory,
                                    paths.morphological_dimension_source_files,
         folder_name + ".zip")
        with ZipFile(zip_file_dir, 'w') as file_dir:
            with tqdm(desc="creating text files per verb", colour="CYAN",
                total=len(list(self.doc_bin.get_docs(self.nlp.vocab)))) as pbar:
                i = 0
                for word in self.verb_dict.keys():

                    pbar.update(1)
                    print(f"processing text file number {i}\n")
                    i += 1
                    for pos in self.spacy_part_of_speech:
                        file = self.create_csv_for_pos(word, pos)
                        if file != "":
                            # add to zip folder
                            file_dir.write(file)
                            # remove from current folder
                            os.remove(file)


    def create_csv_for_pos(self, word, pos)->str:
        if not (self.verb_dict[word][pos]["Instances"]):
            return ""
        if not self.verify_word(word):
            return ""
        p = word + "_" + pos + ".csv"
        with open(p,
                  mode="w", encoding='utf-8', newline="") as f:
            fields = ["lemma", "word form", "sentence",
                      "doc index",
                      "sent index",'token index', 'tokenized sentence']
            d = self.fieldnames_for_csv(fields)
            writer = csv.DictWriter(f=f, fieldnames=fields)
            writer.writerow(d)

            for sent in self.verb_dict[word][pos]["Instances"]:
                n_dict = self.create_instance_row(word, sent)
                writer.writerow(n_dict)
        return p

    """
        create row for zip file for an instance of a word. 
        @:param word(str): lemma of word form
        @:param sent(DictEntry): a DictEntry named tuple representing info
        about the instance
        @:return n_dict(dict): dictionary with all the info
    """
    def create_instance_row(self, word, sent: DictEntry)->dict:
        verb_form, sentence, doc_index, sent_index, token_index, tokenized_sentence = sent
        n_dict = {"lemma": word,
                  "word form": verb_form,
                  "sentence": sentence,
                  "doc index": doc_index,
                  "sent index": sent_index,
                  "token index": token_index,
                  "tokenized sentence": tokenized_sentence}
        return n_dict

    def verify_word(self, word: str)->bool:
        illegal = [':', '*', "?", "<", ">", "|", '"', chr(92), chr(47)]
        return not any(ill in word for ill in illegal)

    """
    check that this token is a verb
    """
    def verify_we_want_to_add_token(self, token:spacy.tokens)->bool:
        return token.pos_ == "VERB"

    """
    create template for dictionary for analyze_verbs function
        Parameters: None
        return: dict(dict) : dictionary for every word
        containe
    """
    def dict_template(self):
        dict = {}
        for pos in self.spacy_part_of_speech:
            dict[pos] = {"lemma": "", "Counter":0, "Instances": set()}
        return dict

    """
    create template for csv shopwing the distribution of tha parts of speech
    """

    def csv_pos_template(self, pos_to_use, types_of_extensions):
        pos_template = []

        for pos in pos_to_use:
            for extension in types_of_extensions:
                if extension != "%":
                    pos_template.append(pos + "_" + extension)
                else:
                    pos_template.append(pos + extension)

        d = {"word": "word"}

        for pos in pos_to_use:
            for extension in types_of_extensions:
                if extension != "%":
                    d[pos + "_" + extension] = pos + "_" + extension
                else:
                    d[pos + extension] = pos + extension


        return pos_template, d


    """
    get fieldnames for csv that represents all sentences with word in a 
    specific pos
    """
    def fieldnames_for_csv(self, given_lst: iter):
        dic = {}
        for fieldname in given_lst:
            dic[fieldname] = fieldname
        return dic





if __name__ == '__main__':
    args = docopt(usage)

    verb_anazlyzer = AnalyzeVerbs(args["<spacy_file_name>"], args["<num_of_posts>"])

    # # write with only open clasue parts of speech and count, %
    # verb_anazlyzer.write_dict_to_csv(pos_to_use=open_class_pos,
    #                                  fields_to_write=["%", "count"], output_file_name=
    #                                  r"open_class_pos.csv"
    #                                  )

    # # write with only count and all parts of speech
    verb_anazlyzer.write_dict_to_csv(pos_to_use=open_class_pos,
                                     fields_to_write=["count", "%"],
                                     output_file_name=r"all_pos_count.csv",
                                     additional_cols=[EXTRA_COLS.PERCENTAGE_AS_OPEN_CLASS_POS,
                                                      EXTRA_COLS.TOTAL_OPEN_CLASS])

    verb_anazlyzer.create_text_files()
