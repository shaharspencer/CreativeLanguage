import copy
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
import spacy
from transformers import pipeline, BertTokenizer, TFBertForMaskedLM

class BertConverter:
    """
    A class for performing predictions using BERT model and POS tagging.

    Attributes:
        tokenizer (BertTokenizer): BERT tokenizer.
        model (TFBertForMaskedLM): BERT model.
        nlp (spacy.Language): Spacy language model.

    Methods:
        get_top_k_predictions: Get top k predictions to replace [MASK] token.
        __append_outputs_to_csv: Append the outputs of BERT predictions to the analyzed dataframe.
        __predict_single_row: Predict replacements and POS tags for a single row.
        __get_top_k_pos_predictions: Get top k POS predictions for a list of replacements.
        __get_top_k_replacements: Get top k replacements for [MASK] token.
        replace_with_token: Replace a token in a tokenized sentence.
    """

    def __init__(self):
        """
        Initialize tokenizer and model instances.
        """
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.model = TFBertForMaskedLM.from_pretrained('bert-base-cased')
        self.nlp = spacy.load("en_core_web_lg")

    def get_top_k_predictions(self, input_dataframe: pd.DataFrame, k=5) -> pd.DataFrame:
        """
        Get top k predictions to replace [MASK] token in the input dataframe.

        Args:
            input_dataframe (pd.DataFrame): Dataframe to get predictions for.
            k (int): Number of predictions to generate (default: 5).

        Returns:
            pd.DataFrame: Dataframe with two columns added - "POS PREDICTIONS" and "BERT REPLACEMENTS".
        """
        replacement_pos_predictions = []
        replacement_word_predictions = []
        for index, row in input_dataframe.iterrows():
            replacements, pos_prediction = self.__predict_single_row(row, k)
            replacement_pos_predictions.append(pos_prediction)
            replacement_word_predictions.append(replacements)
        self.__append_outputs_to_csv(input_dataframe, replacement_pos_predictions, replacement_word_predictions)
        return input_dataframe

    def __append_outputs_to_csv(self, input_dataframe, replacement_pos_predictions: list[str],
                                replacement_word_predictions: list[list[str]]) -> None:
        """
        Add the outputs of the BERT predictions and the new POS prediction to the analyzed dataframe.

        Args:
            input_dataframe: Input dataframe.
            replacement_pos_predictions (list[str]): List of POS predictions.
            replacement_word_predictions (list[list[str]]): List of word replacements.
        """
        input_dataframe["POS PREDICTIONS"] = pd.Series(replacement_pos_predictions)
        input_dataframe["BERT REPLACEMENTS"] = pd.Series(replacement_word_predictions)

    def __predict_single_row(self, row, k):
        """
        Predict replacements and POS tags for a single row.

        Args:
            row: Input row.
            k (int): Number of replacements to generate.

        Returns:
            tuple: A tuple containing replacements and POS predictions.
        """
        row_text = list(copy.deepcopy(row["tokenized sentence"]))
        masked_tokenized_sentence = self.replace_with_token(row_text, row["token index"], replace_with="[MASK]")
        replacements = self.__get_top_k_replacements(" ".join(masked_tokenized_sentence), k=k)
        replacements = replacements.split(" ")
        pos_predictions = self.__get_top_k_pos_predictions(replacements, sentence=row_text, index=row["token index"])
        return replacements, pos_predictions

    def __get_top_k_pos_predictions(self, replacements, sentence, index):
        """
        Get top k POS predictions for a list of replacements.

        Args:
            replacements (list): List of word replacements.
            sentence: Input sentence.
            index (int): Index of the token to replace.

        Returns:
            str: Most frequent POS prediction.
        """
        pos_dict = defaultdict(int)
        for replacement in replacements:
            sent = sentence.copy()
            sent[index] = replacement
            sent = " ".join(sent)
            doc = self.nlp(sent)
            predicted_pos = doc[index].pos_
            pos_dict[predicted_pos] += 1
        return max(pos_dict, key=pos_dict.get)

    def __get_top_k_replacements(self, input_string:str, k):
        """
        Get top k replacements for [MASK] token.

        Args:
            input_string (str): String to replace [MASK] token in.
            k (int): Number of replacements to generate.

        Returns:
            str: Top k replacements.
        """
        tokenized_inputs = self.tokenizer(input_string, return_tensors="tf")
        outputs = self.model(tokenized_inputs["input_ids"])
        top_k_indices = tf.math.top_k(outputs.logits, k).indices[0].numpy()
        decoded_output = self.tokenizer.batch_decode(top_k_indices)
        mask_token = self.tokenizer.encode(self.tokenizer.mask_token)[1:-1]
        mask_index = np.where(tokenized_inputs['input_ids'].numpy()[0] == mask_token)[0][0]
        decoded_output_words = decoded_output[mask_index]
        return decoded_output_words

    def replace_with_token(self, tokenized_sent: list[str], token_index: int, replace_with: str):
        """
        Replace a token in a tokenized sentence.

        Args:
            tokenized_sent (list[str]): Tokenized sentence.
            token_index (int): Index of the token to replace.
            replace_with (str): Replacement token.

        Returns:
            list[str]: Tokenized sentence with the token replaced.
        """
        sent = copy.deepcopy(tokenized_sent)
        sent[token_index] = replace_with
        return sent


if __name__ == '__main__':
    bert = BertConverter()
    dtypes = {
        'lemma': str,
        'word form': str,
        'sentence': str,
        'doc index': int,
        'sent index': int,
        'token index': int,
        'tokenized sentence': str
    }

    df = pd.read_csv(
        r"C:\Users\User\OneDrive\Documents\CreativeLanguageOutputFiles\morphological_dimension\source_files\first_50_posts_with_lg2023_06_17\find_VERB.csv",
        encoding="ISO-8859-1",
        dtype=dtypes, converters={'tokenized sentence': eval},
        on_bad_lines="skip"
    )
    bert.get_top_k_predictions(df)
