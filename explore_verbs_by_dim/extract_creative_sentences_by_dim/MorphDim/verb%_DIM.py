import pandas as pd
import os
import csv
class GetRarestVerbs:
    def __init__(self, sents_dir_path =
    r"C:\Users\User\PycharmProjects\CreativeLanguage\explore_verb_patterns\old_python\create_verb_path_code\verb_sents",
                 verb_csv=
            r"C:\Users\User\PycharmProjects\CreativeLanguage\explore_verb_patterns\extract_creative_sentences\MorphDim\files\verb_path_lemmas_15000.csv",
           ):
        self.verb_csv = verb_csv
        self.sents_dir_path = sents_dir_path

    def explore_simple_method_propn(self,
                                     top_and_lowest_k=50):
        # proper nouns used as verbs

        df = pd.read_csv(self.verb_csv)
        illegal = {':', '*', "?", "<", ">", "|", '"', chr(92), chr(47)}

        sorted_dataframe = df[df['PROPN%'] > 0.8]


        sorted_dataframe = sorted_dataframe.sort_values(by='VERB_count', ascending=True)

        sorted_lowest = sorted_dataframe.head(n=top_and_lowest_k)
        # sorted_top = sorted_dataframe.tail(n=top_and_lowest_k)
        from datetime import datetime
        datetime = datetime.today().strftime('%Y_%m_%d')
        output_path = "morph_order_by_count_after_propn_" + datetime + ".csv"

        self.write_csv_file_from_df(df=sorted_lowest, output_path=output_path)

    def explore_simple_method_by_count(self,
                                       csv_path = r"C:\Users\User\PycharmProjects\CreativeLanguage\explore_verb_patterns\extract_creative_sentences\MorphDim\files\verb_path_lemmas_15000.csv",
                                    top_and_lowest_k=30):
        df = pd.read_csv(csv_path)
        sorted_dataframe =df.sort_values(by='VERB_count', ascending=True)
        # sorted_dataframe = sorted_dataframe[df['PROPN%'] > 0.8 and df["VERB_count"] > 1]
        sorted_lowest = sorted_dataframe.head(n=top_and_lowest_k)
        # sorted_top = sorted_dataframe.tail(n=top_and_lowest_k)
        from datetime import datetime
        datetime = datetime.today().strftime('%Y_%m_%d')
        output_path = "morph_order_by_count_" + datetime + ".csv"
        self.write_csv_file_from_df(df=sorted_lowest,
                                    output_path=output_path)

        # sorted_lowest.to_csv("sorted_by_then_verb_proportion.csv")


    def write_csv_file_from_df(self, output_path, df):
        file = open(output_path, "w", encoding='utf-8', newline='')
        fields = ["lemma", "verb form", "percent as verb", "percent as propn", "Count as verb",
                  "Sentence", "Doc index", "Sent index"]
        writer = csv.DictWriter(f=file, fieldnames=fields)

        d = print_fieldnames(fields)

        writer.writerow(d)

        for index, row in df.iterrows():

            sents_path = os.path.join(self.sents_dir_path, row['word'] + "_VERB.csv")
            # with open(sents_path, encoding='utf-8') as f:
            try:
                sents_df = pd.read_csv(sents_path, encoding='utf-8')
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
            except FileNotFoundError:
                pass
        file.close()




def print_fieldnames(given_lst: iter):
    dic = {}
    for fieldname in given_lst:
        dic[fieldname] = fieldname
    return dic

if __name__ == '__main__':
    obj = GetRarestVerbs()
    obj.explore_simple_method_propn(
    )
    obj.explore_simple_method_by_count()
