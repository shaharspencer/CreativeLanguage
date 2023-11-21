"""
for each sample:
    Decide if we think that spacy is going to make an error (e.g., we think it’s rare or creative)
    if it is:
    run our masking algorithm
    else:
    return spacy’s prediction
"""
import pandas as pd
import spacy
from spacy import tokens
from spacy.tokens import Doc

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
    def __init__(self, rarity_dataframe, masking_k):
        """
        @:param rarity_dataframe (str): helps us deterimne frequencies of token lemmas in corpus
        """
        # open dataframe
        self.rarity_dataframe = self.open_rarity_dataframe(rarity_dataframe)
        # filter dataframe to only include rare verbs tokens
        self.rarity_dataframe = self.filter_rarity_dataframe(self.rarity_dataframe) #TODO
        # self.target_dataframe = "" #TODO

        self.fill_mask = FillMask(top_k=masking_k)

    def run(self, sentence_dataframe, target_dataframe, output_file):
        sentences = self.open_sentences_dataframe(sentence_dataframe)
        target = self.open_target_dataframe(target_dataframe)
        result = self.iterate_over_sentences(sentences_df=sentences, target_df=target)
        result.to_csv(output_file, index=False,
                         encoding='utf-8',
                         columns=['Sentence_Count', 'Token_ID', 'Word',
                                  'UD_POS', 'SPACY_POS', "Mask_Tags",
                                  'Combined_Algorithm_Tags'])

    def open_target_dataframe(self,path_to_file)->pd.DataFrame:
        c_df = pd.read_csv(path_to_file, sep=',',
                           names=['Sentence_Count', 'Token_ID', 'Word',
                                  'UD_POS', 'SPACY_POS'], skiprows=1, index_col=False)
        return c_df

    def open_sentences_dataframe(self, sentences_df_path) -> pd.DataFrame:
        sentences_df = pd.read_csv(sentences_df_path, encoding='utf-8',
                                   header=None)

        # Assign column names manually
        sentences_df.columns = ["sentence"]
        return sentences_df

    def iterate_over_sentences(self, sentences_df:pd.DataFrame,
                               target_df:pd.DataFrame):
        tags,tagged_token = [], []
        # for each row, tokenize the sentence
        for index, row in sentences_df.iterrows():
            # process row, TODO figre how to get rows contents
            tokenized_text = row["sentence"].split()
            row_nlp = nlp(row["sentence"])
            for token in row_nlp:
                # Find the corresponding row in target_df using 'Sentence_Count' and 'Token_ID'
                target_row = target_df[
                    (target_df['Sentence_Count'] == index) &
                    (target_df['Token_ID'] == token.i)
                    ]
                try:
                    if not target_row.iloc[0]['Word'] == token.text:
                        if target_df[
                    (target_df['Sentence_Count'] == index) &
                    (target_df['Token_ID'] == token.i + 1)
                    ]:
                except Exception:
                    z = 0
                tag = self.tag_by_rarity(token=token,
                                         tokenized_sentence_text=tokenized_text)
                tags.append(tag)
                tagged_token.append(token)

        target_df['Combined_Algorithm_Tags'] = tags
        return target_df






    def open_rarity_dataframe(self, dataframe_path)->pd.DataFrame:
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
                      tokenized_sentence_text: list[str])->str:
        """
        This method first checks if the token is rare.
        if so, it tags by the masking algorithm.
        else, it tags by the regular spacy tagger
        """
        if not token.pos_ == "VERB" or not self.check_token_is_rare_from_dataframe(token=token):
            return token.pos_
        # check according to dataframe if token.lemma_ is rare
        else:
            return self.fill_mask.mask(tokenized_text=tokenized_sentence_text,
                                          index=token.i)


if __name__ == '__main__':
    obj = RareTokensAlgorithm(rarity_dataframe=
                             r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\files\source_files\ENSEMBLE_first_40000_posts_openclass_pos_count_2023_08_04.csv",
                             masking_k=1)

    obj.run(sentence_dataframe=r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\files\output_raw_sentences_50000_sentences.csv",
            target_dataframe=r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\files\tags_data\output_with_masked_POS_50000_sentences_top5.csv"
            ,output_file="output.csv")