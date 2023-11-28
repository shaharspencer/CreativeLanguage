import pandas as pd
from docopt import docopt
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

usage = '''
masking_evaluation CLI.
number of 
Usage:
    masking_evaluation.py <data_file>
'''

def evaluate_mask_relative_improvements(csv_path:str = r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\files\tags_data\output_only_mask.csv")->list:
    acuuracies = []
    #open dataframe
    df = pd.read_csv(csv_path, encoding='utf-8')
    #filter dataframe
    filtered_df = df[df['UD_POS'] == df['SPACY_POS']]
    for i in range(1,10):
        acuuracies.append(round(accuracy_score(filtered_df["UD_POS"], filtered_df[f"Only_Mask_Tags_{i}"]), 3))
    print(acuuracies)
    plt.plot(range(1, 10), acuuracies, marker='o', linestyle='-')
    plt.xlabel('Mask Number')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Scores for Masking, on spacy correct preds')
    plt.grid(True)
    plt.show()
def evaluate_verb_accuracy(csv_path=r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\files\tags_data\output_only_mask.csv"):
    acuuracies = []
    # open dataframe
    df = pd.read_csv(csv_path, encoding='utf-8')
    # filter dataframe
    filtered_df = df[df['UD_POS'] == "VERB"]
    for i in range(1, 10):
        acuuracies.append(round(accuracy_score(filtered_df["UD_POS"],
                                               filtered_df[
                                                   f"Only_Mask_Tags_{i}"]), 3))
    print(acuuracies)
    plt.plot(range(1, 10), acuuracies, marker='o', linestyle='-')
    plt.xlabel('Mask Number')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Scores for Masking, on verbs')
    plt.grid(True)
    plt.show()

def mask_accuracy(csv_path=r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\files\tags_data\output_only_mask.csv"):
    acuuracies = []
    # open dataframe
    df = pd.read_csv(csv_path, encoding='utf-8')
    # filter dataframe

    for i in range(1, 10):
        acuuracies.append(round(accuracy_score(df["UD_POS"],
                                               df[
                                                   f"Only_Mask_Tags_{i}"]), 3))
    print(acuuracies)
    plt.plot(range(1, 10), acuuracies, marker='o', linestyle='-')
    plt.xlabel('Mask Number')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Scores for Masking, all tokens')
    plt.grid(True)
    plt.show()





#TODO add CLI
def run(data_file_path):

    #columns=['Sentence_Count', 'Token_ID', 'Word', 'UD_POS', 'SPACY_POS', "Mask_Tags"]
    data = pd.read_csv(data_file_path, encoding='utf-8')

    # filter rows where Mask_Tags is not None
    masked_data = data[data['Mask_Tags'].notnull()]

    accuracy = accuracy_score(masked_data['UD_POS'], masked_data['Mask_Tags'])

    print(f'Accuracy of Mask_Tags compared to UD_POS: {accuracy}%')


if __name__ == '__main__':
    # args = docopt(usage)
    #
    # data_file = args["<data_file>"]
    # if data_file == "None":
    #     run(data_file_path=r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\files\tags_data\output_with_masked_POS_50000_sentences.csv")
    # else:
    #     run(data_file_path=data_file)
    evaluate_mask_relative_improvements()
    evaluate_verb_accuracy()
    mask_accuracy()

