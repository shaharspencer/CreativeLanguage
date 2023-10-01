import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import os

os.environ['HF_DATASETS_CACHE'] = r"C:\Users\User\.cache\huggingface"

import sys

sys.path.append(r"C:\Users\User\PycharmProjects\CreativeLanguage")

from src.source_files_by_dim.dependencies.dependency_list.gpt2_code \
    import GPT2TextGenerator
#TODO need to set order in predictions bc i want to have a set of predictions


from src.generate_and_test_spacy.processors.processor import Processor

nlp = Processor(to_conllu=False, use_ensemble_tagger=True,
                             to_process=False).get_nlp()

class DobjGPTReplacements:
    """
    Class for generating replacements using GPT2 for direct objects (dobj).

    Attributes:
        replacment_generator (GPT2TextGenerator): Instance of GPT2TextGenerator for text generation.
        source_df (pd.DataFrame): DataFrame containing source data.

    Methods:
        open_source_csv: Opens and reads source CSV file.
        generate_replacements: Generates replacement sentences for direct objects.
        truncate_sentence: Truncates sentence until the direct object.
        truncate_sent_until_dobj: Truncates sentence until the direct object based on index.
        process_instance: Processes an instance and generates replacements.
    """
    def __init__(self, source_csv):
        """
         Initializes DobjGPTReplacements class.

         Args:
             source_csv (str): Path to the source CSV file.
         """
        self.replacment_generator = GPT2TextGenerator()
        self.source_df = self.open_source_csv(source_csv=source_csv)


    def open_source_csv(self, source_csv):
        """
        Opens and reads the source CSV file.

        Args:
            source_csv (str): Path to the source CSV file.

        Returns:
            pd.DataFrame: DataFrame containing source data.
        """
        csv = pd.read_csv(source_csv, encoding='utf-8', header=0,
                          names=['lemma (V)', 'sentence', 'verb index',
                                 'verb text', 'dobj', 'dobj index']).head(10)
        return csv


    def generate_replacements(self):
        """
           Generates replacement sentences for direct objects.
           Saves the result to a CSV file.
        """
        self.source_df["truncated sent"] = \
            self.source_df.progress_apply(self.truncate_sentence, axis=1)
        self.source_df['replacement sentences'] = self.source_df.progress_apply\
            (self.process_instance, axis=1)
        self.source_df.to_csv("all_dobj_eat_10_1_2023.csv",
                              encoding='utf-8', index=False, sep=",")

    def truncate_sentence(self, row):
        """
         Truncates the sentence until the direct object.

         Args:
             row (pd.Series): Row of the DataFrame containing 'sentence', 'dobj', and 'dobj index'.

         Returns:
             str: Truncated sentence.
         """
        sentence, dobj, dobj_index = \
            row["sentence"], row["dobj"], row["dobj index"]
        return self.truncate_sent_until_dobj(sentence, dobj, dobj_index)

    def truncate_sent_until_dobj(self, sent_text: str, dobj: str,
                                 dobj_index: int) -> str:
        """
          Truncates the sentence until the direct object based on index.

          Args:
              sent_text (str): Original sentence text.
              dobj (str): Direct object.
              dobj_index (int): Index of the direct object.

          Returns:
              str: Truncated sentence.
        """
        processed_sent = nlp(sent_text)[:(dobj_index + 1)]
        assert processed_sent[-1].text == dobj
        return processed_sent[:-1].text

    def process_instance(self, row):
        """
             Processes an instance and generates replacements.

             Args:
                 row (pd.Series): Row of the DataFrame containing 'truncated sent'.

             Returns:
                 str: Generated replacement sentence.
        """
        truncated_sent, token_index, token_text = row["truncated sent"], row["verb index"], row["verb text"]

        gpt_replace = \
            self.replacment_generator.text_generator_method(truncated_sent, last_token_index=token_index,
                                                            last_token_text=token_text,
                                                            k=5)
        return gpt_replace



if __name__ == '__main__':
    path = r"src\source_files_by_dim\dependencies\dependency_list\eat_dobj_examples_sentences.csv"
    d = DobjGPTReplacements(source_csv=path)
    d.generate_replacements()
