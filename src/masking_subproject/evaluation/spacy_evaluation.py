"""
spacy_evaluation.py

this script evaluates the performance of predicted POS tags against a gold-standard.
It provides functions to open files containing gold-standard and predicted POS tags, and compute
metric such as accuracy.

"""
import os.path

import pandas as pd
from docopt import docopt

usage = '''
spacy_evaluation CLI.
number of 
Usage:
    spacy_evaluation.py
    spacy_evaluation.py <gold_standard_file> <predictions_file>
'''


from sklearn.metrics import accuracy_score

def check_files(df_gold: pd.DataFrame, df_pred: pd.DataFrame) -> bool:
    try:
        assert len(df_gold) == len(df_pred), "Number of lines in gold and predictions files do not match."
        concatenated_tokens = df_gold["Word"] + " " + df_pred["Word"]

        # Create a new column in df_gold indicating whether the tokens match
        df_gold["Tokens_Match"] = concatenated_tokens.str.split().str[0] == \
                                  concatenated_tokens.str.split().str[1]

        assert df_gold["Word"].equals(df_pred["Word"]), "Tokens do not match for corresponding lines."
        assert df_gold["POS_Tag"].equals(df_pred["POS_Tag"])

        print("Files are matching!")
        return True
    except AssertionError as e:
        print(f"AssertionError: {e}")
        mismatch_indices = df_gold.index[df_gold["Word"] != df_pred["Word"]].\
            tolist()
        print(f"Mismatched indices: {mismatch_indices}")
        return False
def open_files(gold_standard_file: str, predictions_file: str) -> \
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # define column names TODO confirm these

    columns = ['Word', 'POS_Tag', 'Sentence_Count', 'Token_ID']

    # read the file into a DataFrame
    df_gold = pd.read_csv(gold_standard_file, sep=',', header=None, names=columns, encoding='utf-8')
    df_pred = pd.read_csv(predictions_file, sep=',', header=None, names=columns, encoding='utf-8')

    check_files(df_gold, df_pred)
    # create a new DataFrame with the specified columns
    df_combined = pd.DataFrame({
        'Sentence_Index': df_gold['Sentence_Count'],
        'Token_Index': df_gold['Token_ID'],
        'Token': df_gold['Word'],
        'UD_POS': df_gold['POS_Tag'],
        'SPACY_POS': df_pred['POS_Tag']
    })

    return df_gold, df_pred, df_combined


def run(n_sentences=50000, gold_standard_file=
        r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\files\tags_data\output_with_pos_UD_tags_10_sentences.txt",
        predictions_file=r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\files\tags_data\output_with_pos_SPACY_tags_10_sentences.txt",
        ) -> str:
    postfix = f"UD_Spacy_combined_tags_{n_sentences}_sentences.csv"
    output_path = os.path.join(r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\files\tags_data",postfix)
    gold_tags, pred_tags, df_combined = open_files(gold_standard_file=gold_standard_file,
                                      predictions_file=predictions_file)

    df_combined.to_csv(output_path, sep=' ', index=False,
                       encoding='utf-8', columns=['Sentence_Index',
        'Token_Index',
        'Token',
        'UD_POS',
        'SPACY_POS'])

    accuracy = accuracy_score(gold_tags["POS_Tag"], pred_tags['POS_Tag'])
    print(f'Accuracy of Spacy_Tags compared to UD_POS: {accuracy}%')
    return output_path


if __name__ == '__main__':
    args = docopt(usage)
    gold_standard = args["<gold_standard_file>"]
    predictions = args["<predictions_file>"]
    if not (gold_standard) and not (predictions):
        run()
    else:
        run(gold_standard_file=gold_standard, predictions_file=predictions)