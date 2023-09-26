import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from src.source_files_by_dim.dependencies.dependency_list.gpt2_code \
    import GPT2TextGenerator
#TODO need to set order in predictions bc i want to have a set of predictions
import os

os.environ['HF_DATASETS_CACHE'] = r"C:\Users\User\.cache\huggingface"

from src.generate_and_test_spacy.processors.processor import Processor

nlp = Processor(to_conllu=False, use_ensemble_tagger=True,
                             to_process=False).get_nlp()

class DobjGPTReplacements:
    def __init__(self, source_csv):
        self.replacment_generator = GPT2TextGenerator()
        self.source_df = self.open_source_csv(source_csv=source_csv)


    def open_source_csv(self, source_csv):
        csv = pd.read_csv(source_csv, encoding='utf-8', header=0,
                          names=['lemma (V)', 'sentence', 'dobj', 'dobj index'])
        return csv


    def generate_replacements(self):
        self.source_df["truncated sent"] = \
            self.source_df.progress_apply(self.truncate_sentence, axis=1)
        self.source_df['replacement sentences'] = self.source_df.progress_apply\
            (self.process_instance, axis=1)
        self.source_df.to_csv("all_dobj_eat.csv",
                              encoding='utf-8', index=False, sep=",")

    def truncate_sentence(self, row):
        sentence, dobj, dobj_index = \
            row["sentence"], row["dobj"], row["dobj index"]
        return self.truncate_sent_until_dobj(sentence, dobj, dobj_index)

    def truncate_sent_until_dobj(self, sent_text: str, dobj: str,
                                 dobj_index: int) -> str:
        processed_sent = nlp(sent_text)[:(dobj_index + 1)]
        assert processed_sent[-1].text == dobj
        return processed_sent[:-1].text

    def process_instance(self, row):
        truncated_sent = row["truncated sent"]

        gpt_replace = \
            self.replacment_generator.text_generator_method(truncated_sent)
        return gpt_replace



if __name__ == '__main__':
    path = r"C:\Users\User\PycharmProjects\CreativeLanguage\src\source_files_by_dim\dependencies\dependency_list\eat_clause_example_yay.csv"
    d = DobjGPTReplacements(source_csv=path)
    d.generate_replacements()
