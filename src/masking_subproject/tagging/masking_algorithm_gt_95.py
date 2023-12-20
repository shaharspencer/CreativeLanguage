

"""
for each sample:
    Decide if we think that spacy is going to make an error (e.g., we think it’s rare or creative)
    if it is:
    run our masking algorithm
    else:
    return spacy’s prediction
"""
import json
import os.path
import sys

import pandas as pd
import spacy
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from spacy import tokens
from spacy.tokens import Doc
from conllu import parse
if spacy.prefer_gpu():
    print("using GPU")
else:
    print("not using gpu")
sys.path.append(r"/cs/snapless/gabis/shaharspencer/CreativeLanguageProject/src/")
sys.path.append('/cs/snapless/gabis/shaharspencer/CreativeLanguageProject/src/')

# h
sys.path.append(r'/cs/snapless/gabis/shaharspencer')
parent_dir = os.path.abspath(r'CreativeLanguageProject/src')

# Append the parent directory to sys.path

sys.path.append(parent_dir)
sys.path.append(r"/cs/snapless/gabis/shaharspencer/CreativeLanguageProject")
# sys.path.append(r"C:\Users\User\PycharmProjects\CreativeLanguage\src")
# sys.path.append(r"C:\Users\User\PycharmProjects\CreativeLanguage")
sys.path.append(r"/cs/snapless/gabis/shaharspencer/CreativeLanguageProject/src/")

from src.masking_subproject.tagging.tag_with_mask import FillMask


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        words = [word for word in words if word.strip()]  # Remove empty tokens
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load("en_core_web_lg")
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


class RareTokensAlgorithm:
    def __init__(self):
        """
        @:param rarity_dataframe (str): helps us deterimne frequencies of token lemmas in corpus
        """
        # open dataframe
        with open(r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\word_frequencies\freq_bands_100.json", "r") as json_file:
            self.rarity_json = json.load(json_file)


    def run(self, target_dataframe, output_file):
        accuracies = []
        k_values = list(range(1, 10))
        spacy_accuracy = accuracy_score(target_dataframe["UD_POS"],
                                        target_dataframe["SPACY_POS"])
        for i in range(1, 10):
            algorithm_i_col = f'algorithm_{i}'
            target_dataframe[algorithm_i_col] = target_dataframe.apply(
                lambda row: row[f'SPACY_POS'] if not self.check_token_is_rare_from_dataframe(
                    row['lemma'], row["SPACY_POS"]) else row[f'Only_Mask_Tags_{i}'], axis=1)
            print(f'accuracy for k = {i} is:')
            print(accuracy_score(target_dataframe["UD_POS"],
                                 target_dataframe[algorithm_i_col]))
            print()
            accuracies.append(accuracy_score(target_dataframe["UD_POS"],
                                 target_dataframe[algorithm_i_col]))
            # Plotting the graph
        plt.figure(figsize=(8, 5))
        plt.plot(k_values, accuracies, marker='o', linestyle='-')
        plt.title('Accuracy of Masking Algorithm for Different k Values\n'
                  'frequency band gt 95\n'
                  'use mask for only Nouns and Propns\n')
        plt.plot(k_values, [spacy_accuracy] * len(k_values), linestyle='--',
                 label='Spacy Prediction')  # Straight line for Spacy's accuracy
        plt.xlabel('k values')
        plt.ylabel('Accuracy')
        plt.xticks(k_values)
        plt.grid(True)
        # Adding labels to each point
        for i, txt in enumerate(accuracies):
            plt.annotate(f'{txt:.6f}', (k_values[i], accuracies[i]),
                         textcoords="offset points", xytext=(0, 5),
                         ha='center')

        plt.annotate(f'{spacy_accuracy:.6f}', (k_values[-1], spacy_accuracy),
                     textcoords="offset points", xytext=(0, 5), ha='center')

        plt.tight_layout()
        plt.show()
        target_dataframe.to_csv(output_file, encoding='utf-8')



    def check_token_is_rare_from_dataframe(self,token_lemma, token_spacy_pos):
        """
        this method recieves a dataframe and a lemmatized token,
        and determines whether this token's lemma is rare
        :param self:
        :return:
   """
        if token_spacy_pos not in ["NOUN", "PROPN"]:
            return False
        try:
            if self.rarity_json[token_spacy_pos][token_lemma] >= 95:
                return True
        except KeyError:
            return False
        return False



if __name__ == '__main__':

        prefix = r"\cs\snapless\gabis\shaharspencer\CreativeLanguageProject\src"
        # prefix = r"C:\Users\User\PycharmProjects\CreativeLanguage\src"
        obj = RareTokensAlgorithm()
        target_dataframe = pd.read_csv(r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\files\tags_data\mask_tags_data\output_only_mask.csv", encoding='utf-8')
        obj.run(output_file=f"output_masking_algorithm_gt_95.csv", target_dataframe=target_dataframe)