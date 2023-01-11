import os.path

import pandas as pd
from docopt import docopt

from general_tests.create_dep_tree import Renderer

usage = '''
check_if_lg_is_good CLI.
Usage:
    check_if_lg_is_good.py [model_name] [csv_path]
'''

def check_if_lg_is_as_good_as_trf(csv_path =
                                  r"C:\Users\User\PycharmProjects\CreativeLanguage\general_tests\dependent_order_by_count_2023_01_01.csv"):

    df = pd.read_csv(csv_path, encoding='utf-8')
    index = 0
    for index, row in df.iterrows():
        output_to_dep_graph(row['Sentence'])
        index += 1
        if index == 5:
            return

def output_to_dep_graph(sentence):
    rend = Renderer("en_core_web_lg")
    path = os.path.join("graph_dir", sentence[:5] +".svg")
    rend.output_sent_to_svg(sentence, path)



if __name__ == '__main__':
    args = docopt(usage)

    if args["model_name"]:
        check_if_lg_is_as_good_as_trf(args["model name"])

    else:
        check_if_lg_is_as_good_as_trf()