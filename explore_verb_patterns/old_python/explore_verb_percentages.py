import pandas as pd
import os

import csv
def explore(csv_path):
    # proper nouns used as verbs
    df = pd.read_csv(csv_path)
    result_df = df[df["VERB%"]< 0.25]
    file = open("verbs_25_perc.txt", "w", encoding='utf-8')
    for index, row in result_df.iterrows():
        file.write(row['word'])
        file.write("\n")

def explore_complex(csv_path):
    # proper nouns used as verbs
    df = pd.read_csv(csv_path)
    result_df = df[df["VERB%"] < 0.01]
    file = open("verbs_1_perc_sents.txt", "w", encoding='utf-8')
    for index, row in result_df.iterrows():
        if os.path.exists(
            r"C:\Users\User\PycharmProjects\indexing_text\verb_path_and_sents\first_15000_posts_with_nbsp\verb_sents/" + row['word'] + "_VERB.txt"):
            with open(r"C:\Users\User\PycharmProjects\indexing_text\verb_path_and_sents\first_15000_posts_with_nbsp\verb_sents/" + row['word'] + "_VERB.txt", "r", encoding='utf-8') as f:
                for line in f:
                    file.write(line)
                    file.write("\n")

    file.close()



if __name__ == '__main__':
    explore_complex(r"C:\Users\User\PycharmProjects\indexing_text\verb_path_and_sents\first_15000_posts_with_nbsp\verb_path_new.csv")