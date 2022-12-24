import pandas as pd
import os
import csv
class GetRarestVerbs:
    def __init__(self):
        pass

    def explore_simple_method(self,
            csv_path
    ,
                              output_path = "rarest_verbs.csv", top_and_lowest_k=20):
        # proper nouns used as verbs

        df = pd.read_csv(csv_path)
        sorted_dataframe = df.sort_values(by='VERB_count', ascending=True)
        sorted_dataframe = sorted_dataframe[df['PROPN%'] > 0.8]
        sorted_lowest = sorted_dataframe.head(n=top_and_lowest_k)
        sorted_top = sorted_dataframe.tail(n=top_and_lowest_k)

        sorted_lowest.to_csv("sorted_by_then_verb_proportion.csv")

        # sort verbs according to frequency as verb
        file = open(output_path, "w", encoding='utf-8')
        # for index, row in result_df.iterrows():
        #     file.write(row['word'])
        #     file.write("\n")



if __name__ == '__main__':
    obj = GetRarestVerbs()
    obj.explore_simple_method(csv_path=r"C:\Users\User\PycharmProjects\CreativeLanguage\extract_creative_sentences\MorphDim\files\verb_path_lemmas_15000.csv",
                                           output_path="RarestAndMostCommon.csv")
