

"""
for each sample:
    Decide if we think that spacy is going to make an error (e.g., we think it’s rare or creative)
    if it is:
    run our masking algorithm
    else:
    return spacy’s prediction
"""
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
sys.path.append(r"C:\Users\User\PycharmProjects\CreativeLanguage\src")
sys.path.append(r"C:\Users\User\PycharmProjects\CreativeLanguage")
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
    def __init__(self, rarity_dataframe):
        """
        @:param rarity_dataframe (str): helps us deterimne frequencies of token lemmas in corpus
        """
        # open dataframe
        self.rarity_dataframe = self.open_rarity_dataframe(rarity_dataframe)
        # filter dataframe to only include rare verbs tokens
        self.rarity_dataframe = self.filter_rarity_dataframe(self.rarity_dataframe) #TODO
        # self.target_dataframe = "" #TODO

    def run(self, conllu_file, target_dataframe, output_file):

        with open(conllu_file, 'r', encoding='utf-8') as conllu_file:
            conllu_content = parse(conllu_file.read())
        target = self.open_target_dataframe(target_dataframe)

        # Define a list to keep track of all new columns added
        all_new_columns = []

        for masking_k in range(1, 11):
            fill_mask = FillMask(top_k=masking_k)
            new_columns = [f"Only_Mask_Tags_{n}" for n in
                           range(1, masking_k + 1)]
            all_new_columns.extend(
                new_columns)  # Add new columns to the tracking list

            result = self.iterate_over_sentences(conllu_content=conllu_content,
                                                 target_df=target,
                                                 fill_mask=fill_mask,
                                                 k=masking_k)

            # Retrieve existing columns in the target DataFrame
            existing_columns = target.columns.tolist()

            # Save all the columns including newly added ones
            columns_to_save = existing_columns + all_new_columns

            # Save the DataFrame to CSV
            result.to_csv(output_file, index=False,
                          encoding='utf-8',
                          columns=columns_to_save)

            # Calculate accuracy
            accuracy = accuracy_score(result['UD_POS'],
                                      result[f'Only_Mask_Tags_{masking_k}'])
            print(
                f'Accuracy of Only Mask tags compared to UD_POS: {accuracy}, k={masking_k}')

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
            for token in row_nlp:

                tag = self.tag_by_rarity(token=token,
                                         tokenized_sentence_text=tokenized_text,
                                         fill_mask=fill_mask)
                tags.append(tag)
                token_list.append(token)
            print(c)
            c+=1

        target_df[f'Only_Mask_Tags_{k}'] = tags
        return target_df

    def open_rarity_dataframe(self, dataframe_path) -> pd.DataFrame:
        dtype_dict = {
            #     'word': str,
            #     'VERB_count': float,
            #     'PROPN_count': float,
            #     'NOUN_count': float,
            #     'ADJ_count': float,
            'total open class': float,
            'VERB%': float,
            'open class pos / total': float
        }
        converters = {'VERB_count': eval}
        df = pd.read_csv(dataframe_path, encoding='ISO-8859-1',
                         converters=converters, dtype=dtype_dict)
        return df

    def filter_rarity_dataframe(self, df):
        df = df[
            (df["VERB%"] < 0.5)
            &
            (df["VERB%"] > 0) &
            (df["open class pos / total"] >= 0.95)
            &
            (df["VERB_count"] <= 5)
            &
            (df["total open class"] > 50)
            ]
        return df

    def check_token_is_rare_from_dataframe(self,token:spacy.tokens):
        """
        this method recieves a dataframe and a lemmatized token,
        and determines whether this token's lemma is rare
        :param self:
        :return:
        """
        # if token.lemma_ is in dataframe's "word" column, return True, else return false
        df = self.rarity_dataframe
        lemma = token.lemma_

        # Check if the lemma exists in the DataFrame's "word" column
        if lemma in df['word'].values:
            return True
        else:
            return False

    def tag_by_rarity(self, token: spacy.tokens,
                      tokenized_sentence_text: list[str], fill_mask: FillMask)->str:
        """
        This method first checks if the token is rare.
        if so, it tags by the masking algorithm.
        else, it tags by the regular spacy tagger
        """
        # if not token.pos_ == "VERB" or not self.check_token_is_rare_from_dataframe(token=token):
        #     return token.pos_
        # # check according to dataframe if token.lemma_ is rare
        # else:
        return fill_mask.predict(tokenized_text=tokenized_sentence_text,
                                          index=token.i)


if __name__ == '__main__':

        prefix = r"\cs\snapless\gabis\shaharspencer\CreativeLanguageProject\src"
        # prefix = r"C:\Users\User\PycharmProjects\CreativeLanguage\src"
        obj = RareTokensAlgorithm(rarity_dataframe=os.path.join(prefix,
                                 r"masking_subproject\files\source_files\ENSEMBLE_first_40000_posts_openclass_pos_count_2023_08_04.csv"))
        obj.run(conllu_file=os.path.join(prefix, r"masking_subproject\files\raw_data\en_ewt-ud-test.conllu"),
                target_dataframe=os.path.join(prefix, r"masking_subproject\files\tags_data\UD_Spacy_combined_tags_50000_sentences.csv")
                ,output_file=f"output_only_mask.csv")