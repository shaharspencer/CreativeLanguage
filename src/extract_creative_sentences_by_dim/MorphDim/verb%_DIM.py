import pandas as pd
import os
import csv
import spacy
from zipfile import ZipFile

from docopt import docopt
from numpy import int32

import src.utils.path_configurations

usage = '''
verb%_DIM CLI.
Usage:
    verb%_DIM.py <sents_dir_path> <verb_csv>
'''


class GetRarestVerbs:
    def __init__(self, sents_dir_path,
                 verb_csv, model = "en_core_web_lg"
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
        self.nlp = spacy.load(model)

    def explore_simple_method_propn(self, output_file_name,
                                    top_and_lowest_k=20):
        # proper nouns used as verbs

        df = pd.read_csv(self.verb_csv)

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
    a specific measure that uses entropy to find creative sentences.
    1. find lemmas that appear as verbs less than max_verb_percentange
    and are over 95% open class.
    2. 
    
    """
    #TODO generalize percentages
    #TODO add index of verb
    def explore_entropy_measures(self, output_file_name,
                                 top_and_lowest_k):
        dtype_dict = {
            'num_POSs': 'float',
            'VERB_count': 'float',
            'PROPN_count': 'float',
            'NOUN_count': 'float',
            'ADJ_count': 'float',
            'total open class': 'float',
            'percent_VERB': 'float',
            'percent_PROPN': 'float',
            'percent_NOUN': 'float',
            'percent_ADJ': 'float',
            'entropy': 'float',
            'open class pos / total': 'float'
        }

        df = pd.read_csv(self.verb_csv, on_bad_lines="skip", encoding="ISO-8859-1",
                         dtype = dtype_dict)
        # filter values by percentage consitions
        df = df[(df["%VERB"] < 0.5)
                &
                (df["%VERB"] > 0)
                &
                (df["open class pos / total"] >= 0.95)
                &
                (df["VERB_count"] <= 5)
                &
                (df["total open class"] > 50)
                ]
        # sort by entropy
        df_sortedby_entropy = df.sort_values(["entropy"], ascending=True)
        self.__write_csv_file_from_df(output_file_name, df_sortedby_entropy)

    """
    this function takes a dataframe with the rarest verbs 
    and write a csv with sentences that have those verbs in them.
    @:param output_path - name and dir of csv
    """

    def __write_csv_file_from_df(self, output_file_path, df):
        output_path = output_file_path
        file = open(output_path, "w", encoding='utf-8', newline='')
        fields = ["lemma", "verb form", "percent as verb", "percent as propn",
                  "Count as verb",
                  "Sentence", "Doc index", "Sent index", "index of verb", "total open class"]
        # if we can add the entropy
        if "entropy" in df.columns:
            fields.append("entropy")
        if "open class pos / total" in df.columns:
            fields.append("%OPENCLASS")
        writer = csv.DictWriter(f=file, fieldnames=fields)

        d = print_fieldnames(fields)
        writer.writerow(d)
        counter = 0
        with ZipFile(self.sents_dir_path, 'r') as zip:
            for index, row in df.iterrows():
                sents_path = row["word"] + "_VERB.csv"
                # with open(sents_path, encoding='utf-8') as f:
                #TODO want more than 30 ofc
                try:
                    sents_df = pd.read_csv(zip.extract(sents_path),
                                           encoding='utf-8')
                    counter += 1
                    if counter == 100:
                        break
                    c = 0
                    for ind, r in sents_df.iterrows():

                        # if c == row["VERB_count"]:
                        #     break
                        n_dict = {"lemma": row['word'],
                                  'verb form': r['word form']}
                        if row["word"] == "omg":
                            t = 0
                        c += 1
                        doc_index = r["doc index"]
                        sent_index = r["sent index"]
                        #TODO issue with my style vs their style - %verb vs verb%
                        #TODO improve this method it looks messy
                        n_dict["percent as verb"] = row['%VERB']
                        n_dict["percent as propn"] = row['%PROPN']
                        n_dict["Sentence"] = r['sentence'].strip()
                        n_dict['Doc index'] = doc_index
                        n_dict['Sent index'] = sent_index
                        n_dict['Count as verb'] = row["VERB_count"]
                        n_dict["total open class"] = row["total open class"]
                        n_dict["index of verb"] = \
                            self.__get_index_of_verb(lemma=row['word'],
                                                     verb_form=r['word form'],
                                                     sent=
                                                     r['sentence'].strip())
                        if "entropy" in fields:
                            n_dict["entropy"] = row["entropy"]
                        if "%OPENCLASS" in fields:
                            n_dict["%OPENCLASS"] = row[
                                "open class pos / total"]


                        writer.writerow(n_dict)
                    os.remove(sents_path)

                except KeyError:
                    pass
            file.close()

    """
      given parameters describing the verb, we find the index spacy gives 
      to the verb within the sentence.
    """
    def __get_index_of_verb(self, lemma: str, verb_form: str, sent: str):
        doc = self.nlp(sent)
        for token in doc:
            if token.text == verb_form and token.lemma_.lower() == lemma:
                return token.i
        return -1


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

    datetime = datetime.today().strftime('%Y_%m_%d')
    # output_path_morph = "morph_order_by_count_" + datetime + ".csv"
    #
    # output_path_count = "morph_order_by_count_after_propn_" + datetime + ".csv"
    #
    # obj.explore_simple_method_propn(output_file_name=output_path_count
    #                                 )
    # obj.explore_simple_method_by_count(
    #     output_file_name=output_path_morph)
    output_path_entropy = "morph_order_by_entropy_and_verb_perc" + datetime + ".csv"
    obj.explore_entropy_measures(output_file_name=output_path_entropy,
                                 top_and_lowest_k=20)
