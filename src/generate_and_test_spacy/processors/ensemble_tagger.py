# for BERT tokenizer
import concurrent
from collections import OrderedDict, Counter

import stanza
import torch
from nltk import map_tag

# for NLTK tokenizer
import nltk

# for flair tokenizer
from flair.data import Sentence
from flair.models import SequenceTagger

pos_tags = [
    'ADJ',
    'ADP',
    'ADV',
    'AUX',
    'CCONJ',
    'DET',
    'INTJ',
    'NOUN',
    'NUM',
    'PART',
    'PRON',
    'PROPN',
    'PUNCT',
    'SCONJ',
    'SYM',
    'VERB',
    'X'
]

nltk_to_spacy_pos_mapping = {
    'VERB': 'VERB',
    'NOUN': 'NOUN',
    'PRON': 'PRON',
    'ADJ': 'ADJ',
    'ADV': 'ADV',
    'ADP': 'ADP',
    'CONJ': 'CCONJ',
    'DET': 'DET',
    'NUM': 'NUM',
    'PRT': 'PART',
    'X': 'X',
    '.': 'PUNCT'
}


class EnsembleTagger:
    def __init__(self):

        # confirm nltk and stanza dependencies
        self.download_stanza_dependencies()
        self.download_nltk_dependencies()

        # define stanza and flair pipelines
        self.stanza_pipeline = stanza.Pipeline('en',
                                               processors='tokenize,mwt,pos',
                                               tokenize_pretokenized=True,
                                               download_method=None)
        # flair dependencies (UD pipeline)
        self.flair_pipeline = SequenceTagger.load("flair/upos-english")

        self.tagger_funcs = [self.flair_tokenizer, self.nltk_tagger,
                             self.stanza_tagger]

    """
        predicts the labels for a sentence based on the majority vote
        of the chosen models. 
        @:param doc 
    """

    def get_tags_list(self, doc):
        word_lst = [token.text for token in doc]
        pos_list = [[token.text, token.pos_] for token in doc]
        votes = self.get_all_votes(spacy_tokens=word_lst, spacy_tags=pos_list)
        tags = self.calculate_votes(votes)
        return tags

    """
        returns dictionary of  majority votes for each token's pos
        dict is of form token_index: majority_tag
        @:param votes list of votes from each tagger 
        @:return dictionary of majority votes
    """

    def calculate_votes(self, votes: [list[list[str, str]]]) -> list:
        polled_results = []
        zipped_lists = zip(*votes)
        for zipped_token in zipped_lists:
            result = self.majority_vote(zipped_token)
            # retrieve results from completed tasks
            polled_results.append(result)
        return polled_results

    """
        this method gets all the votes from the different tokenizers
        for each tagger, we send a list of strings and recieve a list of
        type (token_text, predicted_token_tag)
        @:param spacy_tokens list of strings from spacy's tokenizer
        @:param spacy_tags: spacy's default pos tags for these tokens
        @:return list of lists returned by each tagger
    """

    def get_all_votes(self, spacy_tokens: list[str], spacy_tags: list[list[str, str]]) -> \
    list[list[(str, str)]]:
        tagged_tokens_lst = []
        tagged_tokens_lst.append(spacy_tags)

        for func in self.tagger_funcs:
            result = func(spacy_tokens)
            tagged_tokens_lst.append(result)

        return tagged_tokens_lst

    """
        this method recieves a list of lists, each one is a list returned 
        by a pos tagger
        it returns one list for each token, which consists of tuples of
        type (token_text, predicted_token_tag)
        @:param list_of_lists list from each pos tagger
        @:return list of one list for each token
    """

    def create_lists_from_elements(self, list_of_lists):
        result_lists = []
        transposed = zip(*list_of_lists)
        for elements in transposed:
            result_lists.append(list(elements))

        return result_lists

    """
        finds the tag with maximum votes for some token
        @:param tags potentials tags for some token
        @:return tag the tag that received the majority vote
    """

    def majority_vote(self, tags_and_token)->list[str, str]:
        assert len(tags_and_token) == len(self.tagger_funcs) + 1

        tag_counts = Counter(tag[1] for tag in tags_and_token if tag[1] != 'X')
        if not tag_counts:
            majority_tag = "X"
        else:
            # sort the dictionary so the result is selected consistently
            sorted_items = sorted(tag_counts.items(), key=lambda x: x[1],
                                  reverse=True)
            # only choose majority tag as verb
            # if more than one tagger voted thus
            if sorted_items[0][0] == "VERB" and sorted_items[0][1] <= 1 \
                    and len(sorted_items) > 1:
                majority_tag = sorted_items[1][0]
            else:
                majority_tag = sorted_items[0][0]

        return [tags_and_token[0][0], majority_tag]

    """
          predicts the labels for the tokens based on stanza tokenizer
          @:param doc to tag tokens in 
    """

    def stanza_tagger(self, doc):
        taggings = self.stanza_pipeline([doc])
        tags = [[word.text, word.upos] for word in taggings.sentences[0].words]

        return tags

    """
        predicts the labels for the tokens based on NLTK tokenizer
        @:param doc to tag tokens in 
    """

    def nltk_tagger(self, tokens):
        tagged_tokens = nltk.pos_tag(tokens, tagset='universal')
        mapped = [(word, nltk_to_spacy_pos_mapping[map_tag('en-ptb',
                                                           'universal',
                                                           tag)]) for word,
                                                                      tag in
                  tagged_tokens]
        return mapped

    """
          predicts the labels for the tokens based on flair tokenizer
          @:param doc to tag tokens in 
    """

    def flair_tokenizer(self, token_list: list[str]):
        # since the text is pretokenized, create a sentence object out of it
        sentence = Sentence(token_list)
        self.flair_pipeline.predict(sentence)
        tags = [[token.text, token.labels[0].value] for token in sentence]
        return tags

    def download_stanza_dependencies(self):
        """
        Check if the NLTK dependencies for tokenizer and universal_tagset
        are downloaded, and download them if they are not already present.
        """
        pass
        # stanza.download('en')

    def download_nltk_dependencies(self):
        """
        Check if the NLTK dependencies for tokenizer and universal_tagset
        are downloaded, and download them if they are not already present.
        """
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
            nltk.data.find('taggers/universal_tagset')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
            nltk.download('universal_tagset')

if __name__ == '__main__':
    o = EnsembleTagger()
    o.get_tags_list()