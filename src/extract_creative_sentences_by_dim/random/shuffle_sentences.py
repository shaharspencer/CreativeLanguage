import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.extract_creative_sentences_by_dim.Embeddings import bert_embeddings




class ShuffleSentences:
    def __init__(self):
        self.random_seed = 42
        random.seed(self.random_seed)
        self.embedder = bert_embeddings.ContextualizedEmbeddings()

    def create_embeddings(self, csv_path:str, k:int, output_file_prefix: str):
        """
           Process and embed data for a specified CSV file.

           This method reads a CSV file from the given path and performs the
           following steps:
           1. Subsets k random rows from the dataset.
           2. Splits the subset into training and test sets.
           3. Randomizes test set sentences and calculates indices.
           4. Generates contextualized embeddings for test and train sets.
           5. Outputs embeddings and metadata to files.

           Args:
               csv_path (str): Path to the input CSV file.
               k (int): Number of rows to subset from the dataset.
               output_file_prefix (str): prefix to add to outputs files name.

           Returns:
               None
           """
        d = self.open_csv(csv_path)
        # data_subset = self.get_data_subset(df, k)
        data_subset = d
        train, test = train_test_split(data_subset, test_size=0.2,
                                       random_state=self.random_seed)
        # test = self.randomize_test(test=test)

        test = self.process_and_embed_data(test, randomized=False)
        train = self.process_and_embed_data(train, randomized=False)

        self.output_files(train=train, test=test,
                          output_file_prefix=output_file_prefix, k=k)

    def open_csv(self, csv_path: str) -> pd.DataFrame:
        """
        opens csv with correct datatypes and converters.
        :param csv_path: path to open
        :return: df: opened dataframe with correct dtypes and
                                    converters.
        """
        dtypes = {
            'lemma': str,
            'word form': str,
            'sentence': str,
            'doc index': int,
            'sent index': int,
            'token index': int,
        }
        converters = {'tokenized sentence': eval}
        # open csv
        df = pd.read_csv(csv_path, dtype=dtypes, encoding='utf-8',
                         converters=converters)
        df = self.get_data_subset(df, k=316)
        df_2 = pd.read_csv("embeddings files/earn_VERB.csv", dtype=dtypes, encoding='utf-8',
                           converters=converters)
        df_concat = pd.concat([df, df_2])

        return df_concat

    def output_files(self, train: pd.DataFrame, test: pd.DataFrame,
                     output_file_prefix: str, k: int)-> None:
        """
           Save contextualized embeddings and corresponding metadata to files.

           This function takes two pandas DataFrames representing the training
           and test datasets, concatenates them, and then extracts
           contextualized embeddings and metadata.
           The contextualized embeddings are saved to
           a tab-separated values (tsv) file, and the metadata is
           saved to another tsv file.

           Args:
               train (pd.DataFrame): The training dataset.
               test (pd.DataFrame): The test dataset.
               output_file_prefix (str): prefix to add to outputs file name.
               k (int): subset of how many rows were taken from original df,
                        for unique output file names.

           Returns:
               None
           """
        dataset = pd.concat([test, train])

        contextualized_embeddings = dataset[["contextualized embedding"]]
        filename = output_file_prefix + "_tensor_file_" + str(k) + ".tsv"
        embed_lst = contextualized_embeddings["contextualized embedding"].\
            to_list()
        np.savetxt(filename, embed_lst, delimiter="\t")

        metadata = dataset.drop(columns=["contextualized embedding"])
        self.save_metadata(output_file_prefix, metadata)

    def save_metadata(self, output_file_prefix: str, metadata: pd.DataFrame):
        metadata_file = output_file_prefix + "_tensor_metadata_1000.tsv"
        metadata.to_csv(metadata_file, sep='\t', index=False)


    def randomize_test(self, test: pd.DataFrame) -> pd.DataFrame:
        """
           Randomize sentences in the given test DataFrame and
           calculate related indices.

           This method takes a DataFrame containing test data and
           applies the 'randomize_sentence' function
           to each row, which shuffles the tokenized sentence
            and calculates indices related to the original
           and randomized sentences. The resulting DataFrame
           includes the randomized sentences, shuffled
           indices, and new token indices for each row.

           Args:
               test (pd.DataFrame): Input DataFrame containing test data.

           Returns:
               pd.DataFrame: DataFrame with randomized sentences, shuffled indices,
                             and updated token indices for each row.
        """

        test = test.apply(
            self.randomize_sentence, axis=1)
        return test

    def process_and_embed_data(self, data: pd.DataFrame, randomized: bool) \
            -> pd.DataFrame:
        """
           Process and embed the data in the given DataFrame using contextualized embeddings.

           This method takes a DataFrame containing data to be processed and embedded. It adds a column
           'is_randomized' to indicate whether the data has been randomized. Then, it applies the
           contextualized embedding function to each row in the DataFrame, using either the randomized
           or original token index and tokenized sentence columns based on the 'randomized' parameter.

           Args:
               data (pd.DataFrame): Input DataFrame containing the data to be processed.
               randomized (bool): If True, indicates that the data has been randomized; otherwise, False.

           Returns:
               pd.DataFrame: Processed DataFrame with added contextualized embeddings and metadata columns.
           """
        data["is_randomized"] = randomized
        data = data.apply(self.embedder.contextualized_embeddings,
                          args=("randomized token index" if randomized
                                else "token index",
                                "randomized tokenized sentence" if randomized
                                else "tokenized sentence"),
                          axis=1)
        return data

    def get_data_subset(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        """
            get subset of k random rows from dataset
            @:param df (pd.Dataframe): original dataframe
            @:param k (int): number of rows for the subset
            @:return data_subset (pd.Dataframe): dataframe subsetted from df,
                                                with k rows
        """
        random.seed(self.random_seed)
        random_indices = random.sample(range(len(df)), k)
        data_subset = df.iloc[random_indices]
        return data_subset

    def randomize_sentence(self, row: pd.DataFrame):
        """
          Randomly shuffles the tokenized sentence in the given DataFrame row and
          calculates various indices related to the original and randomized sentences.

          Args:
          row (pd.DataFrame): A DataFrame row containing the following columns:
                              - "tokenized sentence": A list of tokens representing
                                the sentence to be randomized.
                              - "token index": The index of a specific token in the
                                sentence that needs to be tracked.

          Returns:
          pd.DataFrame: The input DataFrame row with the following additional columns:
                        - "randomized indices": A list of indices representing the
                          shuffled order of tokens in the sentence.
                        - "randomized tokenized sentence": A tuple of tokens in the
                          randomized order.
                        - "randomized token index": The new index of the originally
                          specified token in the randomized sentence.
          """
        index_value_pairs = [(index, value) for index, value in
                             enumerate(row["tokenized sentence"])]
        random.shuffle(index_value_pairs)
        row["randomized indices"], \
        row["randomized tokenized sentence"] = zip(*index_value_pairs)
        orig_token_index = row["token index"]
        row["randomized token index"] = \
            row["randomized indices"].index(orig_token_index)

        return row


if __name__ == '__main__':
    s = ShuffleSentences()
    s.create_embeddings("eat_VERB.csv", k=1000, output_file_prefix="eat_earn")
