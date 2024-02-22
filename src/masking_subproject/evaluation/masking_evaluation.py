import json
import random
import uuid

import pandas as pd

from conllu import SentenceList
from docopt import docopt
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

from conllu import parse
i = 0
usage = '''
masking_evaluation CLI.
number of 
Usage:
    masking_evaluation.py <data_file>
'''
def open_conllu(conllu_file_path)-> SentenceList:
    with open(conllu_file_path, 'r', encoding='utf-8') as conllu_file:
        conllu_content = parse(conllu_file.read())
    return conllu_content

def evaluate_mask_relative_improvements(csv_path:str = r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\files\tags_data\mask_tags_data\output_only_mask.csv")->list:
    acuuracies = []
    #open dataframe
    df = pd.read_csv(csv_path, encoding='utf-8')
    #filter dataframe
    # filtered_df = df[df['UD_POS'] != df['SPACY_POS']]
    filtered_df = df
    for i in range(1,10):
        acuuracies.append(round(accuracy_score(filtered_df["UD_POS"], filtered_df[f"Only_Mask_Tags_{i}"]), 3))
    print(acuuracies)
    plt.plot(range(1, 10), acuuracies, marker='o', linestyle='-')
    plt.xlabel('Mask Number')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Scores for Masking, on spacy correct preds')
    plt.grid(True)
    plt.show()
def evaluate_specific_pos_accuracy(pos: str, csv_path=r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\files\tags_data\output_only_mask.csv"):
    acuuracies = []
    # open dataframe
    df = pd.read_csv(csv_path, encoding='utf-8')
    # filter dataframe
    filtered_df = df[df['SPACY_POS'] == pos]
    for i in range(1, 10):
        acuuracies.append(round(accuracy_score(filtered_df["UD_POS"],
                                               filtered_df[
                                                   f"Only_Mask_Tags_{i}"]), 3))
    print(acuuracies)
    spacy_accuracy= accuracy_score(filtered_df["UD_POS"], filtered_df["SPACY_POS"])
    plt.plot(range(1, 10), acuuracies, marker='o', linestyle='-')
    plt.axhline(y=spacy_accuracy, color='r', linestyle='--', label='SpaCy Line')
    plt.text(1, spacy_accuracy + 0.005,
             f'SpaCy Accuracy: {spacy_accuracy:.3f}', color='red')
    plt.xlabel('Mask Number')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Scores for Masking, on pos: {pos}, data points: {len(filtered_df)}')
    plt.grid(True)
    plt.show()

def frequency_band_graphs(pos, frequency_band_json: str, tags_data: str, freq_band_min):
    """
    :param frequency_band_dict:
    :param tags_data:
    :return:
    """
    #open relevant files

    tags_data = pd.read_csv(tags_data, encoding='utf-8')

    tags_data_cols = tags_data.columns
    tags_data = tags_data[tags_data['SPACY_POS'] == pos]
    with open(frequency_band_json, 'r') as json_file:
        frequency_band_dict = json.load(json_file)
    # add bands to data
    band_list = []
    for index, row in tags_data.iterrows():
        try:

            lemma_freq_band = frequency_band_dict[pos][row["lemma"]]
            band_list.append(lemma_freq_band)

        except KeyError:
            band_list.append(-1)

    tags_data["freq_band"] = band_list
    filtered_df = tags_data[tags_data['freq_band'] > 94].copy()

    filtered_df.to_csv(f"rarest_tags_for_{pos}.csv", encoding='utf-8')
    tags_data.to_csv(f"tags_data_pos_{pos}_with_freq_band.csv",
    encoding = 'utf-8')
    # filter df to get the lowest band
    filtered_df = tags_data[((tags_data['freq_band'] >= freq_band_min))]
    # output graph
    acuuracies = []
    for i in range(1, 10):
        acuuracies.append(round(accuracy_score(filtered_df["UD_POS"],
                                               filtered_df[
                                                   f"Only_Mask_Tags_{i}"]), 3))
    print(acuuracies)
    spacy_accuracy = accuracy_score(filtered_df["UD_POS"], filtered_df["SPACY_POS"])
    plt.plot(range(1, 10), acuuracies, marker='o', linestyle='-')
    plt.axhline(y=spacy_accuracy, color='r', linestyle='--', label='SpaCy Line')
    plt.text(1, spacy_accuracy + 0.005,
             f'SpaCy Accuracy: {spacy_accuracy:.3f}', color='red')
    plt.xlabel('Mask Number')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Scores for Masking, on pos: {pos}, data points: {len(filtered_df)} \n frequency band min: {freq_band_min}')
    plt.grid(True)
    # plt.show()
    random_filename = str(uuid.uuid4())
    plt.savefig(f'figs/{pos}_' + random_filename + '.png')
    plt.clf()


def run(data_file_path):

    #columns=['Sentence_Count', 'Token_ID', 'Word', 'UD_POS', 'SPACY_POS', "Mask_Tags"]
    data = pd.read_csv(data_file_path, encoding='utf-8')

    # filter rows where Mask_Tags is not None
    masked_data = data[data['Mask_Tags'].notnull()]

    accuracy = accuracy_score(masked_data['UD_POS'], masked_data['Mask_Tags'])

    print(f'Accuracy of Mask_Tags compared to UD_POS: {accuracy}%')


# def add_lemmas(tags_data, conllu_data_path = r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\files\raw_data\en_ewt-ud-test.conllu"):
#     conllu_data = open_conllu(conllu_data_path)
#     lemma_list = []
#     for sent in conllu_data:
#         for token in sent:
#             if token["xpos"] != None:
#                 lemma_list.append(token["lemma"])
#     tags_data["lemma"] = lemma_list
#     tags_data.to_csv('tags_data.csv', encoding='utf-8')




if __name__ == '__main__':
    # args = docopt(usage)
    #
    # data_file = args["<data_file>"]
    # if data_file == "None":
    #     run(data_file_path=r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\files\tags_data\output_with_masked_POS_50000_sentences.csv")
    # else:
    #     run(data_file_path=data_file)
    evaluate_mask_relative_improvements()
    # pos_list = ["NOUN", "VERB", "ADJ", "ADP", "ADV", "DET", "PROPN" ]
    # for pos in pos_list:
    #     evaluate_specific_pos_accuracy(pos=pos)
    # mask_accuracy()
    #
    # freq_band_json=r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\word_frequencies\freq_bands_100.json"
    # tags_data = r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\files\tags_data\mask_tags_data\output_only_mask.csv"
    # # add_lemmas(pd.read_csv(tags_data, encoding='utf-8'))
    # for i in range(90, 101):
    #     for pos in ["NOUN", "VERB", "ADJ", "ADP", "ADV", "DET", "PROPN" ]:
    #         frequency_band_graphs(pos=pos, frequency_band_json=freq_band_json, tags_data=tags_data, freq_band_min=i)

