import copy
from collections import defaultdict

import numpy as np
import pandas as pd
# from transformers import BertTokenizer, TFBertForMaskedLM
import tensorflow as tf
import spacy
from transformers import pipeline, BertTokenizer, TFBertForMaskedLM

# from src.generate_and_test_spacy.processors import processor
#TODO use multiprocessing



#TODO currenly uses old pos tagger, perhaps will update in the future

#TODO always define data tupes when reading from csv!!!!!!!!!

class BertConverter: # TODO smarter name :)

    """
        initialize tokenizer and model instances
    """
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.model = TFBertForMaskedLM.from_pretrained('bert-base-cased')
        # self.processor = processor.Processor(to_process=False)
        self.nlp = spacy.load("en_core_web_lg")

    """
        returns top k predictions to replace [MASK] token
        receives input of either of a single str or a list[str]
        depending on whether we want to predict for a single sentence 
        or a list of sentences
        @:param input_dataframe(pd.DataFrame): dataframe to get predictions for
        @:return input_dataframe(pf.DataFrame): dataframe with two columns 
                                                added - 
                                                "POS PREDICTIONS",
                                                "BERT REPLACEMENTS"
                                    
    """
    def get_top_k_predictions(self, input_dataframe: pd.DataFrame, k=5) \
                                    -> pd.DataFrame:
        replacement_pos_predictions = []
        replacement_word_predictions = []
        for index, row in input_dataframe.iterrows():
            replacements, pos_prediction = self.__predict_single_row(row, k)
            replacement_pos_predictions.append(pos_prediction)
            replacement_word_predictions.append(replacements)
        self.__append_outputs_to_csv(input_dataframe, replacement_pos_predictions,
                                     replacement_word_predictions)
        return input_dataframe

    """
        adds the outputs of the BERT predictions and the new pos prediction
        to the analyzed dataframe.
        @:param input_dataframe(
    """
    def __append_outputs_to_csv(self, input_dataframe,
                                replacement_pos_predictions: list[str],
                                     replacement_word_predictions: list[list[str]]) -> None:
        input_dataframe["POS PREDICTIONS"] = pd.Series(replacement_pos_predictions)
        input_dataframe["BERT REPLACEMENTS"] = pd.Series(replacement_word_predictions)


    def __predict_single_row(self, row, k):
        row_text = list(copy.deepcopy(row["tokenized sentence"]))
        print(row_text)
        # row_text = [string for string in row_text]
        print(row_text)
        masked_tokenized_sentence = self.replace_with_token(row_text,
                                                            row["token index"],
                                                            replace_with="[MASK]"
                                                            )
        replacements = self.__get_top_k_replacements(
            " ".join(masked_tokenized_sentence), k=k)
        replacements = replacements.split(" ")
        pos_predictions = self.__get_top_k_pos_predictions(replacements,
                                                           sentence=row_text,
                                                           index=row[
                                                               "token index"]
                                                           )
        return replacements, pos_predictions

    """
        
    """
    def __get_top_k_pos_predictions(self, replacements, sentence, index):
        pos_dict = defaultdict(int)
        for replacement in replacements:

            sent = sentence.copy()
            sent[index] = replacement
            sent = " ".join(sent)
            doc = self.nlp(sent)
            predicted_pos = doc[index].pos_
            pos_dict[predicted_pos] += 1

        return max(pos_dict, key=pos_dict.get)



    """
        returns top k replacemnents for [MASK] token.
        @:param input_string(str): string to replace [MASK] token in.
        @:param k(int): how many replacements to generate
        @:return TODO
    """
    def __get_top_k_replacements(self, input_string:str, k):
        tokenized_inputs = self.tokenizer(input_string,
                                          return_tensors="tf")
        outputs = self.model(tokenized_inputs["input_ids"])

        top_k_indices = tf.math.top_k(outputs.logits, k).indices[0].numpy()
        decoded_output = self.tokenizer.batch_decode(top_k_indices)
        mask_token = self.tokenizer.encode(self.tokenizer.mask_token)[1:-1]
        mask_index = \
            np.where(tokenized_inputs['input_ids'].numpy()[0] == mask_token)[0][0]

        decoded_output_words = decoded_output[mask_index]
        return decoded_output_words

    def replace_with_token(self, tokenized_sent: list[str], token_index: int,
                           replace_with: str):
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