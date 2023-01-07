import csv

import pandas as pd


def add_frequency(source_csv = r"C:\Users\User\PycharmProjects\CreativeLanguage\csv_files\verb_path_lemmas_15000.csv", sents_csv =
r"C:\Users\User\Downloads\imtheproblemitsme.xls"):
    source = pd.read_csv(source_csv, encoding='utf-8')
    with open(sents_csv, mode='r', encoding='utf-8'
                                            '') as old_f:
        f = open('csv_file.csv', newline='', encoding ='utf-8', mode=
        'w')

        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file


        # close the file

        sents = pd.read_csv(old_f)
        for ind, row in sents.iterrows():
            quer = row['query']
            relev_row = source.loc[source['word'] == quer]
            writer.writerow(relev_row["VERB%"])

        f.close()


add_frequency()