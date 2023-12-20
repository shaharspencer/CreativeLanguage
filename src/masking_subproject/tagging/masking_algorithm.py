

"""
for each sample:
    Decide if we think that spacy is going to make an error (e.g., we think it’s rare or creative)
    if it is:
    run our masking algorithm
    else:
    return spacy’s prediction
"""
import json
import os.path
import sys

import pandas as pd
import spacy
from sklearn.metrics import accuracy_score
from spacy import tokens
from spacy.tokens import Doc
from conllu import parse
if spacy.prefer_gpu():
    print("using GPU")
else:
    print("not using gpu")
sys.path.append(r"/cs/snapless/gabis/shaharspencer/CreativeLanguageProject/src/")
sys.path.append('/cs/snapless/gabis/shaharspencer/CreativeLanguageProject/src/')

# h
sys.path.append(r'/cs/snapless/gabis/shaharspencer')
parent_dir = os.path.abspath(r'CreativeLanguageProject/src')

# Append the parent directory to sys.path

sys.path.append(parent_dir)
sys.path.append(r"/cs/snapless/gabis/shaharspencer/CreativeLanguageProject")
# sys.path.append(r"C:\Users\User\PycharmProjects\CreativeLanguage\src")
# sys.path.append(r"C:\Users\User\PycharmProjects\CreativeLanguage")
sys.path.append(r"/cs/snapless/gabis/shaharspencer/CreativeLanguageProject/src/")

from src.masking_subproject.tagging.tag_with_mask import FillMask


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        words = [word for word in words if word.strip()]  # Remove empty tokens
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load("en_core_web_lg")
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


class RareTokensAlgorithm:
    def __init__(self):
        """
        @:param rarity_dataframe (str): helps us deterimne frequencies of token lemmas in corpus
        """
        # open dataframe
        with open(r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\word_frequencies\freq_bands_100.json", "r") as json_file:
            self.rarity_json = json.load(json_file)


    def run(self, conllu_file, target_dataframe, output_file):

        with open(conllu_file, 'r', encoding='utf-8') as conllu_file:
            conllu_content = parse(conllu_file.read())
        target = self.open_target_dataframe(target_dataframe)
        print(accuracy_score(target['UD_POS'],
                             target[f'SPACY_POS']))

        # # Define a list to keep track of all new columns added
        # all_new_columns = []

        for masking_k in range(1, 11):
            fill_mask = FillMask(top_k=masking_k)
            result = self.iterate_over_sentences(conllu_content=conllu_content,
                                                 target_df=target,
                                                 fill_mask=fill_mask,
                                                 k=masking_k)

            target = result


            # Calculate accuracy
            accuracy = accuracy_score(result['UD_POS'],
                                      result[f'algorithm_tags_{masking_k}'])
            print(
                f'Accuracy of Algorithm tags compared to UD_POS: {accuracy}, k={masking_k}')



        target.to_csv("algorithm_tags.csv", encoding='utf-8')

    def open_target_dataframe(self,path_to_file)->pd.DataFrame:
        c_df = pd.read_csv(path_to_file, sep=' ',
                           names=['Sentence_Count', 'Token_ID', 'Word',
                                  'UD_POS', 'SPACY_POS'], skiprows=2, index_col=False)
        return c_df



    def iterate_over_sentences(self, conllu_content,
                               target_df: pd.DataFrame, fill_mask: FillMask, k:int):
        tags, token_list = [], []
        c = 0
        for sentence in conllu_content:
            sentence_text = " ".join(
                [str(w) for w in sentence if w["xpos"] != None])

            # process row,
            row_nlp = nlp(sentence_text)
            tokenized_text = sentence_text.split()
            assert len(row_nlp) == len(
                [str(w) for w in sentence if w["xpos"] != None])
            extra_tokens = 0
            for token in sentence:
                token_id = token["id"]
                if token["xpos"] != None:
                    if isinstance(token["id"], tuple):
                        token_id = token_id[0]
                    if not row_nlp[token_id -1].text == sentence[token_id -1 + extra_tokens]["form"]:
                        y = 0
                    tag = self.tag_by_rarity(token=row_nlp[token_id -1],
                                             tokenized_sentence_text=tokenized_text,
                                             fill_mask=fill_mask, token_lemma=sentence[token_id-1 + extra_tokens]["lemma"])
                    tags.append(tag)
                    token_list.append(token)
                else:
                    extra_tokens += 1
            print(c)
            c+=1

        target_df[f'algorithm_tags_{k}'] = tags
        return target_df


    def check_token_is_rare_from_dataframe(self,token_lemma, token:spacy.tokens):
        """
        this method recieves a dataframe and a lemmatized token,
        and determines whether this token's lemma is rare
        :param self:
        :return:
        """
        # if token.lemma_ is in dataframe's "word" column, return True, else return false
        if token.pos_ not in ["NOUN", "PROPN"]:
            return False
        try:
            if self.rarity_json[token.pos_][token_lemma] > 94:
                return True
        except KeyError:
            return False
        return False

    def tag_by_rarity(self, token_lemma, token: spacy.tokens,
                      tokenized_sentence_text: list[str], fill_mask: FillMask)->str:
        """
        This method first checks if the token is rare.
        if so, it tags by the masking algorithm.
        else, it tags by the regular spacy tagger
        """
        if not self.check_token_is_rare_from_dataframe(token=token, token_lemma=token_lemma):
            return token.pos_
        # check according to dataframe if token.lemma_ is rare
        else:
            return fill_mask.predict(tokenized_text=tokenized_sentence_text,
                                          index=token.i)
#TODO some reason iterates a lot further than expected

if __name__ == '__main__':

        prefix = r"\cs\snapless\gabis\shaharspencer\CreativeLanguageProject\src"
        # prefix = r"C:\Users\User\PycharmProjects\CreativeLanguage\src"
        obj = RareTokensAlgorithm()
        obj.run(conllu_file=r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\files\raw_data\en_ewt-ud-test.conllu",
                target_dataframe=r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\files\tags_data\UD_Spacy_combined_tags_50000_sentences.csv"
                ,output_file=f"output_masking_algorithm_gt_95.csv" )