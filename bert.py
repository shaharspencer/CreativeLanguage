import numpy as np
# from transformers import BertTokenizer, TFBertForMaskedLM
import tensorflow as tf
from transformers import pipeline

# TODO change to newer model?

class BertConverter: # TODO smarter name :)

    """
        initialize tokenizer and model instances
    """
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.model = TFBertForMaskedLM.from_pretrained('bert-base-cased')

    """
        returns top k predictions to replace [MASK] token
        receives input of either of a single str or a list[str]
        depending on whether we want to predict for a single sentence 
        or a list of sentences
        @:param input: either str or list[str] 
        @:return decoded output words
    """
    def get_top_k_predictions(self, input_string, k=5) -> str: #TODO decide what the input is and adjust accordingly
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


if __name__ == '__main__':
    bert = BertConverter()