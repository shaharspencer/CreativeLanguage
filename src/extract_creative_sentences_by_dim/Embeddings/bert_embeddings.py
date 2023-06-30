from transformers import pipeline, AutoTokenizer

from transformers import RobertaTokenizer
from transformers import RobertaModel
import torch

class ContextualizedEmbeddings:

        def __init__(self):
            """
               Initializes an instance of the ContextualizedEmbeddings
               class. The constructor loads the pre-trained 'roberta-base'
               model and tokenizer.
            """
            model_name = 'roberta-base'
            self.__model = RobertaModel.from_pretrained(model_name)
            self.__tokenizer = RobertaTokenizer.from_pretrained(model_name)

        def contextualized_embeddings(self, tokenized_text: tuple, verb_index)\
                ->torch.tensor:
            """
                   Returns the contextualized embeddings for a specific verb
                   in a tokenized sentence.
                   Args:
                       tokenized_text (tuple): The tokenized text.
                       verb_index (int): The index representing the position
                       of the verb in the tokenized text.
                   Returns:
                       torch.tensor: The embeddings of the verb as a tensor.
                   Raises:
                       IndexError: If the verb_index is out of range for the
                        tokenized_text.

            """
            input_ids = self.__tokenizer.convert_tokens_to_ids(tokenized_text)
            input_ids_tensor = torch.tensor(
                [input_ids]) #TODO not list?
            outputs = self.__model(input_ids_tensor)
            embeddings = outputs.last_hidden_state[0][verb_index]
            return embeddings


if __name__ == '__main__':
    # Example usage
    obj = ContextualizedEmbeddings()
    text = ("i", "love", "you")
    obj.contextualized_embeddings(text, 2)
