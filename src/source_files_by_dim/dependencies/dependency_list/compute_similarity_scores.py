from typing import List, Tuple

import docopt
import numpy as np
import pandas as pd
from similarity_score_methods import SimilarityScore
from tqdm import tqdm
tqdm.pandas()


usage = '''
compute_similarity_scores.py CLI.
Usage:
    compute_similarity_scores.py <file_to_process> 
'''

from src.generate_and_test_spacy.processors.processor import Processor

nlp = Processor(to_conllu=False, use_ensemble_tagger=True,
                             to_process=False).get_nlp()
class ComputeSimilarity:
    """
         A class for computing similarity scores between tokens and sentences
         on a particular csv file.
    """
    def __init__(self, csv_name):
        """
             Initialize the ComputeSimilarity object.

             Args:
                 csv_name (str): The name of the CSV file containing data.
        """
        self.similarity_scorer = SimilarityScore(nlp=nlp)
        self.source_df = self.open_source_csv(csv_name)

    def open_source_csv(self, source_csv) -> pd.DataFrame:
        """
              Open and read data from a CSV file.

              Args:
                  source_csv (str): The path to the source CSV file.

              Returns:
                  pd.DataFrame: A DataFrame containing the data from the CSV file.
        """
        try:
            converters = {'replacement sentences': eval}
            csv = pd.read_csv(source_csv, encoding='ISO-8859-1', header=0,
                              names=['lemma (V)', 'sentence',
                                     'verb index', 'verb text',
                                     'dobj',
                                     'dobj index', "truncated sent",
                                     'replacement sentences'],
                              converters=converters)
            return csv
        except FileNotFoundError as e:
            raise e
        except Exception as e:
            raise Exception(f"An error occurred while reading the CSV file: {e}")

    def output_to_csv(self, date: str):
        """
             Save the DataFrame to a CSV file.
        """
        self.source_df.to_csv(date + "all_dobj_eat_similarity_scores.csv",
                              encoding='utf-8', index=False, sep=",")

    def get_all_sim_scores(self):
        """
            Calculate both token and sentence similarity scores.
        """
        self.get_all_token_similarity_scores()
        self.get_all_sentence_similarity_scores()

    def get_all_token_similarity_scores(self):
        """
             Compute similarity scores between all the predicted
             tokens and the original token, and add column with list of tuples
             (generated_token, similarity_score) where similarity_score
             is between the original token and the generated one.
        """
        self.source_df["all token similarity scores"] = self.source_df.progress_apply(
            self.calculate_token_similarity_scores, axis=1)

    def get_all_sentence_similarity_scores(self):
        """
             Compute similarity scores between all the predicted
             sentences and the original sentence, and add column with list of tuples
             (generated_sent, similarity_score) where similarity_score
             is between the original sent and the generated one.
        """

        self.source_df["all sentence similarity scores"] = self.source_df.\
            progress_apply(
            self.calculate_sentence_similarity_scores, axis=1)

    def calculate_sentence_similarity_scores(self, row) -> List[Tuple[str, float]]:
        """
           Compute similarity scores between all generated sentences.
           Args:
               row (pd.Series): A row from the source DataFrame.
           Returns:
               list: A list of similarity scores between sentences.
        """
        sent_row_scores = []

        original_sentence = row["truncated sent"] + " " + row["dobj"]
        for sent in row["replacement sentences"]:
            sent_row_scores.append((sent, self.similarity_scorer.
            sentence_pair_similarity(
                sent_1=sent, sent_2=original_sentence)))
        return sent_row_scores

    def calculate_token_similarity_scores(self, row) -> List[Tuple[str, float]]:
        """
         Compute similarity scores between all generated tokens,
         which are just the last token in each generated sentence,
         since they are both a noun and a dobj child of the verb.
         Args:
             row (pd.Series): A row from the source DataFrame.
         Returns:
             list: A list of similarity scores between tokens.
         """
        token_row_scores = []

        original_token = row["dobj"]

        for sent in row["replacement sentences"]:
            gen_token = nlp(sent)[-1].text
            sim_score = self.similarity_scorer.token_pair_similarity(
                gen_token, original_token)
            token_row_scores.append((gen_token, sim_score))
        return token_row_scores

    def get_df_mean_sim_score(self):
        """
               Calculate the mean similarity score for the entire
                df for both sentences and tokens.
        """
        self.source_df["mean sent score"] = self.source_df.progress_apply(
            self.get_row_mean_sent_sim_score, axis=1)
        self.source_df["mean token score"] = self.source_df.progress_apply(
            self.get_row_mean_token_sim_score, axis=1)

    def get_row_mean_sent_sim_score(self, row) -> float:
        """
               Calculate the mean sentence similarity score for a row.

               Args:
                   row (pd.Series): A row from the source DataFrame.

               Returns:
                   float: The mean sentence similarity score.
        """
        all_scores = row["all sentence similarity scores"]
        if not all_scores:
            return -1
        mean_scores = sum(score[1] for score in all_scores) / len(all_scores)
        return mean_scores

    def get_row_mean_token_sim_score(self, row) -> float:
        """
              Calculate the mean token similarity score for a row.
              Args:
                  row (pd.Series): A row from the source DataFrame.
              Returns:
                  float: The mean token similarity score.
        """
        all_scores = row["all token similarity scores"]
        if not all_scores:
            return -1
        sum_scores = sum(score[1] for score in all_scores) / len(all_scores)
        return sum_scores

    def get_df_median_sim_score(self):
        """
            Calculate the median similarity scores for both sentences and tokens
            for the entire df.
        """
        self.source_df["median sent score"] = self.source_df.progress_apply(
            self.get_row_median_sent_sim_score, axis=1
        )
        self.source_df["median token score"] = self.source_df.progress_apply(
            self.get_row_median_token_sim_score, axis=1)

    def get_row_median_sent_sim_score(self, row) -> float:
        """
            Calculate the median sentence similarity score for a row.

            Args:
                row (pd.Series): A row from the source DataFrame.

            Returns:
                float: The median sentence similarity score.
        """
        all_scores = row["all sentence similarity scores"]
        if not all_scores:
            return -1
        median_score = np.median([score[1] for score in all_scores])
        return median_score

    def get_row_median_token_sim_score(self, row) -> float:
        """
          Calculate the median token similarity score for a row.

          Args:
              row (pd.Series): A row from the source DataFrame.

          Returns:
              float: The median token similarity score.
          """
        all_scores = row["all token similarity scores"]
        if not all_scores:
            return -1
        median_score = np.median([score[1] for score in all_scores])
        return median_score

    def get_df_max_sim_score(self):
        """
            Calculate the maximum similarity scores for both sentences and tokens
            for the entire df.
        """
        self.source_df["max sent score"] = \
            self.source_df.progress_apply(
            self.row_max_sent_sim_scores, axis=1
        )
        self.source_df["max token score"] = \
            self.source_df.progress_apply(
            self.row_max_token_sim_scores, axis=1
        )

    def row_max_token_sim_scores(self, row) -> float:
        """
         Calculate the maximum token similarity score for a row.

         Args:
             row (pd.Series): A row from the source DataFrame.

         Returns:
             float: The maximum token similarity score.
         """
        max_token, max_score = None, -1
        all_scores = row["all token similarity scores"]

        for token, score in all_scores:
            if max_score < score:
                max_token, max_score = token, score

        return max_score

    def row_max_sent_sim_scores(self, row) -> float:
        """
        Calculate the maximum sentence similarity score for a row.

        Args:
            row (pd.Series): A row from the source DataFrame.

        Returns:
            float: The maximum sentence similarity score.
        """
        max_sent, max_score = None, -1
        all_scores = row["all sentence similarity scores"]

        for sent, score in all_scores:
            if max_score < score:
                max_sent, max_score = sent, score

        return max_score



if __name__ == '__main__':
    import datetime
    # get current date
    current_date = datetime.date.today()
    formatted_date = current_date.strftime('%d_%m_%y')
    args = docopt.docopt(usage)
    file_name = args["<file_to_process>"]
    obj = ComputeSimilarity(csv_name=file_name)

    obj.get_all_sim_scores()

    obj.get_df_max_sim_score()
    obj.get_df_median_sim_score()
    obj.get_df_mean_sim_score()

    obj.output_to_csv(date=formatted_date)
