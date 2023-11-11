"""
spacy_evaluation.py

this script evaluates the performance of predicted POS tags against a gold-standard.
It provides functions to open files containing gold-standard and predicted POS tags, and compute
metric such as accuracy.

"""
from docopt import docopt

usage = '''
spacy_evaluation CLI.
number of 
Usage:
    spacy_evaluation.py
    spacy_evaluation.py <gold_standard_file> <predictions_file>
'''


from sklearn.metrics import accuracy_score

def check_files(gold_tags, pred_tags):
    assert len(gold_tags) == len(
        pred_tags), "Number of lines in gold and predictions files do not match."

    for gold_line, pred_line in zip(gold_tags, pred_tags):
        gold_prefix = gold_line.split()[0]
        pred_prefix = pred_line.split()[0]
        assert gold_prefix == pred_prefix, "Sentence prefixes do not match for corresponding lines."

    unique_gold_tags = set(gold_tags)
    unique_pred_tags = set(pred_tags)
    assert unique_pred_tags == unique_gold_tags, "UD and Spacy tags do not match."

def open_files(gold_standard_file: str, predictions_file: str) -> tuple:

    with open(gold_standard_file, "r", encoding="utf-8") as f:
        gold_tags = [line.split()[1] for line in f if line.strip()]

    with open(predictions_file, "r", encoding="utf-8") as f:
        pred_tags = [line.split()[1] for line in f if line.strip()]

    check_files(gold_tags=gold_tags, pred_tags=pred_tags)

    return gold_tags, pred_tags


def run(gold_standard_file=
        r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\files\tags_data\output_with_pos_UD_tags_10_sentences.txt",
        predictions_file=r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\files\tags_data\output_with_pos_SPACY_tags_10_sentences.txt") -> float:
    gold_tags, pred_tags = open_files(gold_standard_file=gold_standard_file,
                                      predictions_file=predictions_file)
    accuracy = accuracy_score(gold_tags, pred_tags)
    return accuracy


if __name__ == '__main__':
    args = docopt(usage)
    gold_standard = args["<gold_standard_file>"]
    predictions = args["<predictions_file>"]
    if not (gold_standard) and not (predictions):
        run()
    else:
        run(gold_standard_file=gold_standard, predictions_file=predictions)