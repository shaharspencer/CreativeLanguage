import numpy as np
import pandas as pd
import os
import csv
# import spacy
from zipfile import ZipFile

from docopt import docopt

import src.utils.path_configurations
from scipy.stats import entropy

from src.extract_creative_sentences_by_dim.Embeddings.bert import FillMask

#TODO when saving to csv, save as utf-8
usage = '''
verb%_DIM CLI.
Usage:
    verb%_DIM.py <sents_dir_path> <verb_csv> <tagger>
'''



class GetRarestVerbs:
    def __init__(self, sents_dir_path,
                 verb_csv
                 ):
        self.verb_csv = os.path.join(
            src.utils.path_configurations.files_directory,
            src.utils.path_configurations.morphological_dimension_directory,
            src.utils.path_configurations.morphological_dimension_source_files,
            verb_csv)
        self.sents_dir_path = os.path.join(
            src.utils.path_configurations.files_directory,
            src.utils.path_configurations.morphological_dimension_directory,
            src.utils.path_configurations.morphological_dimension_source_files,
            sents_dir_path)

    def explore_simple_method_propn(self, output_file_name,
                                    top_and_lowest_k=20):
        # proper nouns used as verbs

        df = pd.read_csv(self.verb_csv, encoding='utf-8')

        sorted_dataframe = df[df['PROPN%'] > 0.8]

        sorted_dataframe = sorted_dataframe.sort_values(
            ['VERB_count', "VERB%"],
            ascending=[True, True])

        sorted_lowest = sorted_dataframe.head(n=top_and_lowest_k)
        # sorted_top = sorted_dataframe.tail(n=top_and_lowest_k)
        output_path = os.path.join(
            src.utils.path_configurations.files_directory,
            src.utils.path_configurations.rare_sents_directory,
            output_file_name)

        self.write_csv_file_from_df(df=sorted_lowest,
                                    output_file_path=output_path)

    def explore_simple_method_by_count(self,
                                       output_file_name,
                                       top_and_lowest_k=50):
        df = pd.read_csv(self.verb_csv)
        sorted_dataframe = df.sort_values(['VERB_count', "VERB%"],
                                          ascending=[True, True])

        sorted_lowest = sorted_dataframe.head(n=top_and_lowest_k)
        # sorted_top = sorted_dataframe.tail(n=top_and_lowest_k)
        output_path = os.path.join(
            src.utils.path_configurations.files_directory,
            src.utils.path_configurations.rare_sents_directory,
            output_file_name)

        self.write_csv_file_from_df(df=sorted_lowest,
                                    output_file_path=output_path)

        # sorted_lowest.to_csv("sorted_by_then_verb_proportion.csv")

    """
        this method adds an "Entropy" column to each row in a dataframe
        representing the entropy for that row
        @:param data(pd.DataFrame): dataframe to analyze
    """
    def add_entropy_column(self, data: pd.DataFrame)-> None:

        counts = data[
            ['VERB_count', 'PROPN_count', 'NOUN_count', 'ADJ_count']].to_numpy(
            dtype=np.float64)

        # calculate the entropy for each row
        entropies = []
        for i in range(counts.shape[0]):
            row_entropy = entropy(counts[i], base=2)
            entropies.append(row_entropy)
        data["Entropy"] = entropies


    """
    a specific measure that uses entropy to find creative sentences.
    1. find lemmas that appear as verbs less than max_verb_percentange
    and are over 95% open class.
    2. 
    @param tagger(str):
                                 either REGULAR or ENSEMBLE:
                                whether we want to use a regular spacy tagger
                                or the ensemble tagger
    
    """
    #TODO generalize percentages
    #TODO add index of verb
    def explore_entropy_measures(self, output_file_name,
                                 tagger)->None:
        dtype_dict = {
        #     'word': str,
            'VERB_count': float,
        #     'PROPN_count': float,
        #     'NOUN_count': float,
        #     'ADJ_count': float,
            'total open class': float,
            'VERB%': float,
            'Entropy': float,
            'open class pos / total': float
        }
        converters = {'VERB_count': eval}
        df = pd.read_csv(self.verb_csv, encoding='ISO-8859-1',
                       converters=converters, dtype=dtype_dict)
        self.add_entropy_column(df)
        # filter values by percentage conditions

        df = df[
                 (df["VERB%"] < 0.5)
                 &
                (df["VERB%"] > 0) &
                (df["open class pos / total"] >= 0.95)
                &
                (df["VERB_count"] <= 5)
                &
                (df["total open class"] > 50)
                ]
        # sort by entropy
        df_sortedby_entropy = df.sort_values(["Entropy"], ascending=True)
        self.__write_csv_file_from_df(output_file_name, df_sortedby_entropy,
                                      tagger)

    """
    this function takes a dataframe with the rarest verbs 
    and write a csv with sentences that have those verbs in them.
    @:param output_path - name and dir of csv
    @:param df(pd.DataFrame): source dataframe with info about verbs
    @:param tagger(str): either REGULAR or ENSEMBLE:
                                whether we want to use a regular spacy tagger
                                or the ensemble tagger
    """

    def __write_csv_file_from_df(self, output_file_path, df, tagger:str)->None:

        bert = FillMask()
        output_path = output_file_path
        file = open(output_path, "w", encoding='utf-8', newline='')
        fields = self.__define_fields()
        writer = csv.DictWriter(f=file, fieldnames=fields)

        d = print_fieldnames(fields)
        writer.writerow(d)
        counter = 0
        dtypes = {
        'lemma': str,
        'word form': str,
        'sentence': str,
        'doc index': int,
        'sent index': int,
        'token index': int,
        'tokenized sentence': str
        }
        converters = {'tokenized sentence': eval}
        with ZipFile(self.sents_dir_path, 'r') as zip:
            for index, row in df.iterrows():
                sents_path = row["word"] + "_VERB.csv"

                try:

                    sents_df = pd.read_csv(zip.extract(sents_path),
                                           encoding='utf-8', dtype=dtypes,
                                           converters=converters,
                                           on_bad_lines='skip')

                    # get bert predictions

                    bert.get_top_k_predictions(sents_df, tagger)

                    counter += 1
                    print(f"chosen sentence number {counter}\n")
                    if counter == 100:
                        break
                    for ind, r in sents_df.iterrows():

                        n_dict = self.__define_ndict(r=r, row=row,
                                                     fields=fields)

                        writer.writerow(n_dict)
                    os.remove(sents_path)

                except KeyError:
                    print("key error!\n")
            file.close()
    """
        define column names in output csv
        @:return fields(list): list of strings defining column names
    """
    def __define_fields(self):
        fields = ["lemma", "verb form", "percent as verb", "percent as propn",
                  "Count as verb",
                  "Sentence", "Doc index", "Sent index", "index of verb",
                  "tokenized sentence", "total open class", "BERT tag",
                  "BERT replacements"]
        # if we can add the entropy
        # if "Entropy" in df.columns:
        fields.append("entropy")
        # if "open class pos / total" in df.columns:
        fields.append("%OPENCLASS")
        return fields
    """
        for an instance of a verb, defines a dictionary to write to
        the creative sentence file
        @:param r
        @:param row
        @:param fields TODO define these
    """
    def __define_ndict(self, r, row, fields):
        n_dict = {"lemma": row['word'],
                  'verb form': r['word form']}
        # extract info about lemma
        n_dict["percent as verb"] = row['VERB%']
        n_dict["percent as propn"] = row['PROPN%']
        n_dict['Count as verb'] = row["VERB_count"]
        n_dict["total open class"] = row["total open class"]
        if "entropy" in fields:
            n_dict["entropy"] = row["Entropy"]
        if "%OPENCLASS" in fields:
            n_dict["%OPENCLASS"] = row[
                "open class pos / total"]

        # extract info about specific instance
        n_dict["Sentence"] = r['sentence'].strip()
        doc_index = r["doc index"]
        sent_index = r["sent index"]
        n_dict['Doc index'] = doc_index
        n_dict['Sent index'] = sent_index
        n_dict["BERT tag"] = r["POS PREDICTIONS"]
        n_dict["BERT replacements"] = r["BERT REPLACEMENTS"]

        n_dict["index of verb"] = r["token index"]
        n_dict["tokenized sentence"] = r["tokenized sentence"]


        return n_dict


def print_fieldnames(given_lst: iter):
    dic = {}
    for fieldname in given_lst:
        dic[fieldname] = fieldname
    return dic


if __name__ == '__main__':
    args = docopt(usage)

    obj = GetRarestVerbs(args["<sents_dir_path>"],
                         args["<verb_csv>"])

    from datetime import datetime
    tagger = args["<tagger>"]
    datetime = datetime.today().strftime('%Y_%m_%d')
    # output_path_morph = "morph_order_by_count_" + datetime + ".csv"
    #
    # output_path_count = "morph_order_by_count_after_propn_" + datetime + ".csv"
    #
    # obj.explore_simple_method_propn(output_file_name=output_path_count
    #                                 )
    # obj.explore_simple_method_by_count(
    #     output_file_name=output_path_morph)
    # obj.add_entropy_column()
    output_path_entropy = "morph_order_by_entropy_and_verb_perc" + datetime + ".csv"
    obj.explore_entropy_measures(output_file_name=output_path_entropy,
                                 tagger=tagger)
