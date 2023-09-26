from transformers import pipeline, set_seed

import os

os.environ['HF_DATASETS_CACHE'] = r"C:\Users\User\.cache\huggingface"

from src.generate_and_test_spacy.processors.processor import Processor

nlp = Processor(to_conllu=False, use_ensemble_tagger=True,
                             to_process=False).get_nlp()

def is_noun_prediction(g):
    return nlp(g)[-1].pos_ == "NOUN"

class GPT2TextGenerator:
    """
        A class for generating text using the GPT-2 model.

        Attributes:
            generator (transformers.pipelines.TextGenerationPipeline):
            A pipeline for text generation using GPT-2.
    """

    def __init__(self):
        """
               Initializes the GPT2TextGenerator class by
               setting up the text generation pipeline and seed.
        """
        self.generator = pipeline('text-generation', model='gpt2-large',
                                  temperature=0.7)


        set_seed(42)




    def text_generator_method(self, sent_to_complete,
                              filter_func=is_noun_prediction, k=1):
        """
            Generates text based on the given input.

            Args:
                sent_to_complete (str): The input text to be completed.

            Returns:
                list of str: A list of generated text sequences.
        """
        generated_texts = set()
        ret = self.generator(sent_to_complete, max_new_tokens=1,
                             num_return_sequences=15)
        for item in ret:
            # if we have enough predictions, break
            if len(generated_texts) >= k:
                break
            # else check if this is the type of prediction we want & add
            if filter_func(item["generated_text"]):
                generated_texts.add(item["generated_text"])

        return generated_texts




if __name__ == '__main__':
    gpt = GPT2TextGenerator()
    gpt.text_generator_method("I am eating a", is_noun_prediction)
