import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModel

from transformers import RobertaTokenizer
from transformers import RobertaModel
import torch



#C:\Users\User\anaconda3\envs\myCreativeEnv\python.exe C:\Users\User\PycharmProjects\CreativeLanguageWithVenv\src\extract_creative_sentences_by_dim\Embeddings\bert_embeddings.py
# Some weights of the model checkpoint at roberta-base were not used when initializing RobertaModel: ['lm_head.layer_norm.bias', 'lm_head.layer_norm.weight', 'lm_head.dense.bias', 'lm_head.bias', 'lm_head.dense.weight']
# - This IS expected if you are initializing RobertaModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing RobertaModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
# Some weights of RobertaModel were not initialized from the model checkpoint at roberta-base and are newly initialized: ['roberta.pooler.dense.weight', 'roberta.pooler.dense.bias']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

import csv


class ContextualizedEmbeddings:

        def __init__(self):
            """
               Initializes an instance of the ContextualizedEmbeddings
               class. The constructor loads the pre-trained 'roberta-base'
               model and tokenizer.
            """
            model_name = 'roberta-base'
            self.__model = AutoModel.from_pretrained(model_name)
            self.__tokenizer = AutoTokenizer.from_pretrained(model_name)

        def process_csv(self, csv)->pd.DataFrame:
            """
               Processes the CSV data to compute contextualized embeddings for each row.

               Args:
                   csv (pd.DataFrame): The input DataFrame containing the data.

               Returns:
                   pd.DataFrame: The updated DataFrame with the computed embeddings.
            """
            embeddings = []
            for _, row in csv.iterrows():
                tokenized_sentence, verb_index = row["tokenized sentence"], row["token index"]
                e = self.contextualized_embeddings(tokenized_text=tokenized_sentence,
                                                   verb_index=verb_index)
                embeddings.append(pd.Series(e.detach().numpy()))
            csv["verb embeddings"] = pd.Series(embeddings)
            return csv




        def contextualized_embeddings(self, tokenized_text: tuple, verb_index: int)\
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
            input_ids_tensor = torch.tensor([input_ids])
            outputs = self.__model(input_ids_tensor)
            if verb_index >= len(outputs.last_hidden_state[0]):
                return torch.zeros(768)
            embeddings = outputs.last_hidden_state[0][verb_index]
            return embeddings


if __name__ == '__main__':
    # Example usage
    # obj = ContextualizedEmbeddings()
    # text = ("i", "love", "you")
    # obj.contextualized_embeddings(text, 2)
    dtypes = {
        'lemma': str,
        'word form': str,
        'sentence': str,
        'doc index': int,
        'sent index': int,
        'token index': int,
    }
    converters = {'tokenized sentence': eval}
    c = pd.read_csv("graduate_VERB.csv", encoding='ISO-8859-1', dtype=dtypes,
                    converters=converters)

    d = ContextualizedEmbeddings().process_csv(c)

    # faiss = FAISS(d)



