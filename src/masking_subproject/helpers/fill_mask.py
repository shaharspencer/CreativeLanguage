import copy

import spacy
from transformers import pipeline


class FillMask:


        def predict_tag(self, tokenized_sentence, token_index) \
                -> tuple[list[str], str]:
            # try:
            masked_tokenized_sentence = self.replace_with_token(tokenized_sentence,
                                                                token_index,
                                                                replace_with="<mask>"
                                                                )

            replacements = self.__get_top_k_replacements(
                " ".join(masked_tokenized_sentence))
            x=0
            # pos_predictions = self.__get_top_k_pos_predictions(replacements,sentence=row_text)
            # return replacements, pos_predictions


        """
            do pos prediction for each replacement string.
            do this using either the ensemble_tagger or the regular spacy tagger
            depending on how the sentence was initially analyzed.
            @:param replacements(list[str]): list of potential replacements
            @:param sentence(list[str]): tokenized sentence
            @:param index(int): index of word to replace
            @:param tagger(str): either REGULAR or ENSEMBLE:
                                    whether we want to use a regular spacy tagger
                                    or the ensemble tagger
            @:return majority pos prediction for replacement
        
        """




        """
            returns top k replacemnents for [MASK] token.
            @:param input_string(str): string to replace [MASK] token in.
            @:param k(int): how many replacements to generate
            @:return TODO
        """

        def __get_top_k_replacements(self, input_string: str) -> list[str]:
            replacements = self.classifier(input_string)
            replacement_tokens = [entry["token_str"] for entry in replacements]
            return replacement_tokens

        def replace_with_token(self, tokenized_sent: list[str],
                               token_index: int,
                               replace_with: str) -> list[str]:
            sent = copy.deepcopy(tokenized_sent)
            if token_index >= len(sent):
                raise Exception("list index out of range")
            sent[token_index] = replace_with
            return sent


if __name__ == '__main__':
    f = FillMask()
    f.predict_tag(["I", "love", "you"], token_index=1)