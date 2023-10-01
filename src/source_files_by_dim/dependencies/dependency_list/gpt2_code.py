from transformers import pipeline, set_seed

import os

os.environ['HF_DATASETS_CACHE'] = r"C:\Users\User\.cache\huggingface"

from src.generate_and_test_spacy.processors.processor import Processor

nlp = Processor(to_conllu=False, use_ensemble_tagger=True,
                             to_process=False).get_nlp()

def truncate_noun_dobj_pred(g, last_token_index, last_token_text):
    """
     Args:
         g (str): Generated text.

     Returns:
         bool: True if the last token is a noun, False otherwise.
     """
    tokenized_text = nlp(g)
    if len(tokenized_text) < last_token_index:
        return None
    assert tokenized_text[last_token_index].text == last_token_text
    for child in tokenized_text[last_token_index].children:
        if child.dep_ == "dobj" and child.pos_ == "NOUN"\
                and child.i > last_token_index:
            r = tokenized_text[:child.i + 1].text
            if r == 'Who wants to huddle together underground eating canned food and drinking beer':
                x = 0
            return r

    return None



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
                              last_token_index,
                              last_token_text,
                              filter_func=truncate_noun_dobj_pred,
                              k=1) -> list[str]:
        """
         Generates text based on the given input.

         Args:
             sent_to_complete (str): The input text to be completed.
             filter_func (function, optional): A function to filter the generated text.
                 Defaults to is_noun_prediction.
             k (int, optional): The number of generated texts to return. Defaults to 1.

         Returns:
             list of str: A list of generated text sequences.
         """
        print(sent_to_complete)
        generated_texts = []
        ret = self.generator(sent_to_complete, max_new_tokens=6,
                             num_return_sequences=15)
        for item in ret:
            # if we have enough predictions, break
            if len(generated_texts) >= k:
                break
            # else check if this is the type of prediction we want & add
            truncated_sent = filter_func(item["generated_text"],
                           last_token_index, last_token_text)
            if (truncated_sent) and not ( truncated_sent in generated_texts):
                generated_texts.append(truncated_sent)

        return generated_texts




if __name__ == '__main__':
    gpt = GPT2TextGenerator()
    gpt.text_generator_method(sent_to_complete="I am eating a",
                              filter_func=truncate_noun_dobj_pred,
                              last_token_index=2,
                              last_token_text="eating", k=4)
