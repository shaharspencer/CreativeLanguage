import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
tqdm.pandas()

from src.generate_and_test_spacy.processors.processor import Processor

nlp = Processor(to_conllu=False, use_ensemble_tagger=True,
                             to_process=False).get_nlp()

class ComputeSimilarity:
    """
             A class for computing similarity scores between tokens and sentences.
    """
    def __init__(self, csv_name):
        """
             Initialize the ComputeSimilarity object.

             Args:
                 csv_name (str): The name of the CSV file containing data.
        """
        self.source_df = self.open_source_csv(csv_name)
        self.sentence_model = SentenceTransformer(
            'sentence-transformers/all-mpnet-base-v2')

    def open_source_csv(self, source_csv):  # TODO datatypes
        """
              Open and read data from a CSV file.

              Args:
                  source_csv (str): The path to the source CSV file.

              Returns:
                  pd.DataFrame: A DataFrame containing the data from the CSV file.
        """

        converters = {'replacement sentences': eval}
        csv = pd.read_csv(source_csv, encoding='utf-8', header=0,
                          names=['lemma (V)', 'sentence', 'dobj',
                                 'dobj index', "truncated sent",
                                 'replacement sentences'],
                          converters=converters)
        return csv

    def output_to_csv(self):
        """
             Save the DataFrame to a CSV file.
        """
        self.source_df.to_csv("all_dobj_eat_similarity_scores.csv",
                              encoding='utf-8', index=False, sep=",")


    def compute_token_similarity_scores(self): #TODO create generic naming scheme
        """
             Compute similarity scores between the first of the predicted
             tokens and the original token, and add column with similarity
             score between them
        """
        self.source_df["token similarity"] = self.source_df.progress_apply(
            self.first_token_pair, axis=1)

    def first_token_pair(self, row) -> float:
        """
           Compute similarity between the first replacement token and
           the original token.

           Args:
               row (pd.Series): A row from the DataFrame.

           Returns:
               float: The similarity score.
        """
        if not row["replacement sentences"]:
            return -1
        first_sent = next(iter(row["replacement sentences"]))
        first_replacement_token = nlp(first_sent)[-1].text
        original_token = row["dobj"]
        return self.token_pair_similarity(first_replacement_token, original_token)

    def token_pair_similarity(self, first_token, second_token):
        """
              Compute similarity between two tokens.

              Args:
                  first_token (str): The first token.
                  second_token (str): The second token.

              Returns:
                  float: The similarity score.
            """
        spacy_token_1, spacy_token_2 = nlp(first_token), nlp(second_token)
        return spacy_token_1.similarity(spacy_token_2)


    def compute_sentence_similarity_scores(self): #TODO create generic naming scheme
        """
           Compute similarity scores between the first of the predicted
           sequences and the original sequence, and add column with similarity
           score between them
        """
        self.source_df["sentence similarity"] = self.source_df.progress_apply(
            self.first_sentence_pair, axis=1)

    def first_sentence_pair(self, row):
        """
          Compute similarity between the first replacement sentence and the original sentence.

          Args:
              row (pd.Series): A row from the DataFrame.

          Returns:
              float: The similarity score.
        """
        if not row["replacement sentences"]:
            return -1
        first_sent = next(iter(row["replacement sentences"]))
        original_sentence = row["truncated sent"] + " " + row["dobj"] #TODO this is kind of a raw fix that might only work for nouns
        return self.sentence_pair_similarity(sent_1=first_sent,
                                             sent_2=original_sentence)

    def sentence_pair_similarity(self, sent_1, sent_2):
        """
        Compute similarity between two sentences.

        Args:
            sent_1 (str): The first sentence.
            sent_2 (str): The second sentence.

        Returns:
            float: The similarity score.
        """
        embedding_1 = self.sentence_model.encode(sent_1,
                                                 convert_to_tensor=True)
        embedding_2 = self.sentence_model.encode(sent_2,
                                                 convert_to_tensor=True)

        sim = util.pytorch_cos_sim(embedding_1, embedding_2)[0]
        return sim[0].item()


if __name__ == '__main__':
    path = r"C:\Users\User\PycharmProjects\CreativeLanguage\src\source_files_by_dim\dependencies\all_dobj_eat.csv"
    obj = ComputeSimilarity(csv_name=path)
    obj.compute_sentence_similarity_scores()
    obj.compute_token_similarity_scores()
    obj.output_to_csv()
