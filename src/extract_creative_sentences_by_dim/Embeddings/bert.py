import copy
from collections import defaultdict
import numpy as np
import pandas as pd
import tensorflow as tf
import spacy
from transformers import DebertaTokenizer, DebertaModel
from transformers import pipeline
from src.generate_and_test_spacy.processors.processor import Processor


# TODO use multiprocessing


class FillMask:
    """
        initialize tokenizer and model instances
    """

    def __init__(self):
        # self.tokenizer = DebertaTokenizer.from_pretrained(
        #     "microsoft/deberta-v3-base")
        # self.model = DebertaModel.from_pretrained(
        #     "microsoft/deberta-v3-base")

        self.classifier = pipeline("fill-mask", "xlm-roberta-large", top_k=5)
        self.nlp = spacy.load("en_core_web_lg")
        self.processor = Processor(to_process=False, to_conllu=False)

        text = "I love my <mask>."

        predicted_tokens = self.classifier(text)
        for token in predicted_tokens:
            print(token)

    """
        returns top k predictions to replace [MASK] token
        receives input of either of a single str or a list[str]
        depending on whether we want to predict for a single sentence 
        or a list of sentences
        @:param input_dataframe(pd.DataFrame): dataframe to get predictions for
        @:param tagger(str): whether to use regular spaCy tagger or 
                                        ensemble_tagger
        @:return input_dataframe(pf.DataFrame): dataframe with two columns 
                                                added - 
                                                "ROBERTA POS PREDICTIONS",
                                                "ROBERTA REPLACEMENTS"
    """

    def get_top_k_predictions(self, input_dataframe: pd.DataFrame, tagger) \
            -> pd.DataFrame:
        replacement_pos_predictions = []
        replacement_word_predictions = []
        for index, row in input_dataframe.iterrows():
            replacements, pos_prediction = self.__predict_single_row(row,
                                                                     tagger)
            replacement_pos_predictions.append(pos_prediction)
            replacement_word_predictions.append(replacements)
        self.__append_outputs_to_csv(input_dataframe,
                                     replacement_pos_predictions,
                                     replacement_word_predictions)
        return input_dataframe

    """
        adds the outputs of the ROBERTA predictions and the new pos prediction
        to the analyzed dataframe.
        @:param input_dataframe(
    """

    def __append_outputs_to_csv(self, input_dataframe,
                                replacement_pos_predictions: list[str],
                                replacement_word_predictions: list[
                                    list[str]]) -> None:
        input_dataframe["POS PREDICTIONS"] = pd.Series(
            replacement_pos_predictions)
        input_dataframe["ROBERTA REPLACEMENTS"] = pd.Series(
            replacement_word_predictions)

    """
        return tuple of potential replacements and predicted ensemble pos.
        @:param row(pd.Series): row to predict for
        @:param k(int): how many replacements to generate
        @:param tagger(str):   either REGULAR or ENSEMBLE:
                                whether we want to use a regular spacy tagger
                                or the ensemble tagger
        @:return replacements, pos_predictions(tuple[list[str], str]):
                    potential replacements and pos ensemble prediction on 
                    replacements

    """

    def __predict_single_row(self, row: pd.Series, tagger: str) \
            -> tuple[list[str], str]:
        row_text = list(copy.deepcopy(row["tokenized sentence"]))
        try:
            masked_tokenized_sentence = self.replace_with_token(row_text,
                                                                row[
                                                                    "index of verb"],
                                                                replace_with="<mask>"
                                                                )
        except Exception:
            return ([""], "")
        replacements = self.__get_top_k_replacements(
            " ".join(masked_tokenized_sentence))
        pos_predictions = self.__get_top_k_pos_predictions(replacements,
                                                           tagger=tagger,
                                                           sentence=row_text,
                                                           index=row[
                                                               "index of verb"]
                                                           )
        return replacements, pos_predictions

    """
        do pos prediction for each replacement string.
        do this using either the ensemble_tagger or the regular spacy tagger
        depending on how the sentence was initially analyzed.
        @:param replacements(list[str]): list of potential replacements
        @:param sentence(list[str]): tokenized sentence
        @:param index(int): index of word to replace
        @:param tagger(str): either REGULAR or ENSEMBLE:
                                whether we want to use a regular spacy tagger
                                or the ensemble tagger
        @:return majority pos prediction for replacement

    """

    def __get_top_k_pos_predictions(self, replacements, sentence, index,
                                    tagger) -> str:
        pos_dict = defaultdict(int)
        for replacement in replacements:

            sent = sentence.copy()
            sent[index] = replacement
            sent = " ".join(sent)
            if tagger == "REGULAR":
                doc = self.nlp(sent)
                predicted_pos = doc[index].pos_
                pos_dict[predicted_pos] += 1
            elif tagger == "ENSEMBLE":
                doc = self.processor.process_text(sent)
                predicted_pos = doc[index].pos_
                pos_dict[predicted_pos] += 1
            else:
                raise Exception("tagger choice is not usable\n")

        return max(pos_dict, key=pos_dict.get)

    """
        returns top k replacemnents for [MASK] token.
        @:param input_string(str): string to replace [MASK] token in.
        @:param k(int): how many replacements to generate
        @:return TODO
    """

    def __get_top_k_replacements(self, input_string: str) -> list[str]:
        replacements = self.classifier(input_string)
        replacement_tokens = [entry["token_str"] for entry in replacements]
        return replacement_tokens

    def replace_with_token(self, tokenized_sent: list[str], token_index: int,
                           replace_with: str) -> list[str]:
        sent = copy.deepcopy(tokenized_sent)
        if token_index >= len(sent):
            raise Exception("list index out of range")
        sent[token_index] = replace_with
        return sent


if __name__ == '__main__':
    fill_mask = FillMask()

    dtypes = {
        'lemma': str,
        'word form': str,
        'sentence': str,
        'doc index': int,
        'sent index': int,
        'index of verb': int,
        'tokenized sentence': str
    }

    df_list = [
        r"morph_order_by_entropy_and_verb_perc2023_06_19_with_BERT.csv"]

    # opened_df_zero = pd.read_csv(df_list[0],
    # encoding="ISO-8859-1",
    #     dtype=dtypes, converters={'tokenized sentence': eval},
    #     on_bad_lines="skip"
    #  )
    #
    # new_df_ensemble = fill_mask.get_top_k_predictions(opened_df_zero,"ENSEMBLE")
    #
    # new_df_ensemble.to_csv("roberta_ensembletagger.csv")

    opened_df_one = pd.read_csv(df_list[0],
                                encoding="ISO-8859-1",
                                dtype=dtypes,
                                converters={'tokenized sentence': eval},
                                on_bad_lines="skip"
                                )

    new_df_regular = fill_mask.get_top_k_predictions(opened_df_one, "REGULAR")
    new_df_regular.to_csv("roberta_regulartagger.csv")