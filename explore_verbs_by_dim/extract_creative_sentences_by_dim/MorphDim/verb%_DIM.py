import pandas as pd
import os
import csv

from zipfile import ZipFile
class GetRarestVerbs:
    def __init__(self, sents_dir_path = r"C:\Users\User\OneDrive\Documents\CreativeLanguageOutputFiles\morphological_dimension\source_files\pos_distribution_sentences\first_30000_posts_with_lg.zip",
                 verb_csv=
            r"C:\Users\User\OneDrive\Documents\CreativeLanguageOutputFiles\morphological_dimension\source_files\pos_distribution_csv\first_30000_lg_model.csv",
           ):
        self.verb_csv = verb_csv
        self.sents_dir_path = sents_dir_path

    def explore_simple_method_propn(self, output_dir, output_file_name,
                                     top_and_lowest_k=50):
        # proper nouns used as verbs

        df = pd.read_csv(self.verb_csv)
        illegal = {':', '*', "?", "<", ">", "|", '"', chr(92), chr(47)}

        sorted_dataframe = df[df['PROPN%'] > 0.8]


        sorted_dataframe = sorted_dataframe.sort_values(by='VERB_count',
                                                        ascending=True)

        sorted_lowest = sorted_dataframe.head(n=top_and_lowest_k)
        # sorted_top = sorted_dataframe.tail(n=top_and_lowest_k)


        self.write_csv_file_from_df(df=sorted_lowest, output_dir=output_dir,
                                    output_file_name=output_file_name)

    def explore_simple_method_by_count(self,
                                       output_dir, output_file_name,
                                    top_and_lowest_k=30):
        df = pd.read_csv(self.verb_csv)
        sorted_dataframe =df.sort_values(by='VERB_count', ascending=True)
        # sorted_dataframe = sorted_dataframe[df['PROPN%'] > 0.8 and df["VERB_count"] > 1]
        sorted_lowest = sorted_dataframe.head(n=top_and_lowest_k)
        # sorted_top = sorted_dataframe.tail(n=top_and_lowest_k)

        self.write_csv_file_from_df(df=sorted_lowest,
                                    output_dir=output_dir, output_file_name = output_file_name)

        # sorted_lowest.to_csv("sorted_by_then_verb_proportion.csv")

    """
    this function takes a dataframe with the rarest verbs 
    and write a csv with sentences that have those verbs in them.
    @:param output_path - name and dir of csv
    """
    def write_csv_file_from_df(self, output_dir, output_file_name, df):
        output_path = os.path.join(output_dir, output_file_name)
        file = open(output_path, "w", encoding='utf-8', newline='')
        fields = ["lemma", "verb form", "percent as verb", "percent as propn", "Count as verb",
                  "Sentence", "Doc index", "Sent index"]
        writer = csv.DictWriter(f=file, fieldnames=fields)

        d = print_fieldnames(fields)

        writer.writerow(d)
        with ZipFile(self.sents_dir_path, 'r') as zip:
            for index, row in df.iterrows():
                sents_path = os.path.join(self.sents_dir_path, row['word'] + "_VERB.csv")
                # with open(sents_path, encoding='utf-8') as f:
                try:
                    sents_df = pd.read_csv(zip.extract(sents_path), encoding='utf-8')
                    for ind, r in sents_df.iterrows():
                            n_dict = {"lemma": row['word'], 'verb form': r['word form']}
                            doc_index = r["doc index"]
                            sent_index = r["sent index"]
                            n_dict["percent as verb"] = row['VERB%']
                            n_dict["percent as propn"] = row['PROPN%']
                            n_dict["Sentence"] = r['sentence'].strip()
                            n_dict['Doc index'] = doc_index
                            n_dict['Sent index'] = sent_index
                            n_dict['Count as verb'] = row['VERB_count']
                            writer.writerow(n_dict)

                except KeyError:
                    pass
            file.close()



def print_fieldnames(given_lst: iter):
    dic = {}
    for fieldname in given_lst:
        dic[fieldname] = fieldname
    return dic

if __name__ == '__main__':

    from datetime import datetime
    datetime = datetime.today().strftime('%Y_%m_%d')
    output_path_morph = "morph_order_by_count_" + datetime + ".csv"
    output_dir = r"C:\Users\User\OneDrive\Documents\CreativeLanguageOutputFiles\rarest_sentences_by_dim"

    output_path_count = "morph_order_by_count_after_propn_" + datetime + ".csv"
    obj = GetRarestVerbs()
    obj.explore_simple_method_propn(output_dir=output_dir, output_file_name=output_path_count
    )
    obj.explore_simple_method_by_count(output_dir=output_dir,
                                       output_file_name=output_path_morph)
