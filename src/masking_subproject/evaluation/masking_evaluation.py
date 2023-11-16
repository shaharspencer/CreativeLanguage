import pandas as pd
from docopt import docopt
from sklearn.metrics import accuracy_score

usage = '''
masking_evaluation CLI.
number of 
Usage:
    masking_evaluation.py <data_file>
'''

#TODO add CLI
def run(data_file_path):

    #columns=['Sentence_Count', 'Token_ID', 'Word', 'UD_POS', 'SPACY_POS', "Mask_Tags"]
    data = pd.read_csv(data_file_path, encoding='utf-8')

    # filter rows where Mask_Tags is not None
    masked_data = data[data['Mask_Tags'].notnull()]

    accuracy = accuracy_score(masked_data['UD_POS'], masked_data['Mask_Tags'])

    print(f'Accuracy of Mask_Tags compared to UD_POS: {accuracy}%')


if __name__ == '__main__':
    args = docopt(usage)

    data_file = args["<data_file>"]
    if data_file == "None":
        run(data_file_path=r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\files\tags_data\output_with_masked_POS_50000_sentences.csv")
    else:
        run(data_file_path=data_file)

