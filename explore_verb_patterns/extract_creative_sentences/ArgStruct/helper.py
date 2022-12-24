import pandas as pd
import csv
def helper(csv_path=r"C:\Users\User\PycharmProjects\CreativeLanguage\explore_verb_patterns\Dependencies\first_15000_posts_sents_arg_struct_dim.csv", output_path = "hi.csv"):
    dic = dict()
    for indx, row in pd.read_csv(csv_path, encoding = 'utf-8').iterrows():
        dic[row['Dep struct']] = dic.get(row['Dep struct'], 0) + 1

    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["dep_struct","count"])
        for key in dic:
            writer.writerow([key, dic[key]])


helper()