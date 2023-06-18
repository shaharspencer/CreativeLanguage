from collections import defaultdict

import numpy as np
import pandas as pd
# from transformers import BertTokenizer, TFBertForMaskedLM
import tensorflow as tf
from transformers import pipeline, BertTokenizer, TFBertForMaskedLM

from src.generate_and_test_spacy.processors import processor


# TODO change to newer model?

class BertConverter: # TODO smarter name :)

    """
        initialize tokenizer and model instances
    """
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.model = TFBertForMaskedLM.from_pretrained('bert-base-cased')
        self.processor = processor.Processor(to_process=False)

    """
        returns top k predictions to replace [MASK] token
        receives input of either of a single str or a list[str]
        depending on whether we want to predict for a single sentence 
        or a list of sentences
        @:param input: either str or list[str] 
        @:return decoded output words
    """
    def get_top_k_predictions(self, input_dataframe, k=5) -> str: #TODO decide what the input is and adjust accordingly
        for index, row in input_dataframe.iterrows():
            row_text = row["tokenized sentence"]
            masked_tokenized_sentence = self.replace_with_token(row_text,
                                                               row["token index"],
                                                               replace_with = "[MASK]"
                                                               )
            replacements = self.__get_top_k_replacements(" ".join(masked_tokenized_sentence), k=k)
            pos_predictions = self.__get_top_k_pos_predictions(replacements,
                                                               sentence = row["tokenized sentence"],
                                                               index=row["token index"]
                                                               )
            y = 0


    """
        
    """
    def __get_top_k_pos_predictions(self, replacements, sentence, index):
        pos_dict = defaultdict(str)
        for replacement in replacements:

            sent = sentence.copy()
            sent[index] = replacement
            sent = "".join(sent)
            doc = self.processor.process_text(sent)
            # predicted_pos = doc.sents[0].tokens[index].pos_
            x = 0

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

    def replace_with_token(self, tokenized_sent: tuple[str], token_index: int,
                           replace_with: str):
        sent = tokenized_sent.copy()
        sent[token_index] = replace_with
        return sent


if __name__ == '__main__':
    bert = BertConverter()
    df = pd.read_csv(
        r"C:\Users\User\OneDrive\Documents\CreativeLanguageOutputFiles\morphological_dimension\source_files\first_50_posts_with_lg2023_06_17\find_VERB.csv",
    encoding="ISO-8859-1",
        on_bad_lines="skip"
    )
    bert.get_top_k_predictions(df)