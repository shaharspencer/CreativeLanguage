import datasets
import numpy as np
import pandas as pd
from datasets import Dataset, Features, Value

from transformers import pipeline, AutoTokenizer, AutoModel, \
    RobertaTokenizerFast

#TODO length of input ids is weird!!!!!!!!!!!!
from transformers import RobertaTokenizer
from transformers import RobertaModel
import torch

#TODO time efficiency!!!!!!!!!!!1

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
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            model_name = 'roberta-base'

            self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name,
                                                    add_prefix_space=True)
            self.__model = AutoModel.from_pretrained(model_name).to(self.device)
            # self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        def process_dataset(self, dataset: Dataset)->Dataset:
            """
               Processes the CSV data to compute contextualized embeddings for each row.

               Args:
                   csv (pd.DataFrame): The input DataFrame containing the data.

               Returns:
                   pd.DataFrame: The updated DataFrame with the computed embeddings.
            """

            embeddings_dataset = dataset.map(self.contextualized_embeddings, batched=True)

            filename = "celebrate_tensor_file.tsv"
            embed_lst = embeddings_dataset["context embedding"]
            np.savetxt(filename, embed_lst, delimiter="\t")

            return embeddings_dataset




        @torch.no_grad()
        def contextualized_embeddings(self, row)\
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
            # TODO bug???

            tokenized_sent = row["tokenized sentence"]
            token_index = row["token index"]
            input_ids_tensors = self.tokenizer.batch_encode_plus(tokenized_sent,
                                                         is_split_into_words=
                                                         True, return_tensors="pt",
                                                         padding=True)["input_ids"]
            # input_ids_tensor = torch.tensor(input_ids, device=self.device)
            # with torch.no_grad():
            outputs = self.__model(input_ids_tensors)
            # outputs = self.__model(input_ids_tensor) #TODO test this
            embeddings = outputs.last_hidden_state[0][[index + 1 for index in token_index]].  \
                detach().numpy()
            row["context embedding"] = embeddings
            return row


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
    c = pd.read_csv("graduate_VERB.csv", encoding='utf-8', dtype=dtypes,
                    converters=converters)
    # features = datasets.Features({'lemma': Value('string'),
    #     'word form': Value('string'),
    #     'sentence': str,
    #     'doc index': int,
    #     'sent index': int,
    #     'token index': int,
    #                               'tokenized sentence': eval     })
    # csv_filename = "see_VERB_meta_file.csv"
    c = Dataset.from_pandas(c)
    # c.to_csv(csv_filename, sep='\t', index=False)
    #
    d = ContextualizedEmbeddings().process_dataset(c)




