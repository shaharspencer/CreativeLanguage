# for BERT tokenizer
import concurrent
import stanza
from nltk import map_tag

from spacy import Language
import torch
import spacy
from spacy.tokens import Doc

# to execute code concurrently
from concurrent.futures import ThreadPoolExecutor

# for NLTK tokenizer
import nltk

# for flair tokenizer
import flair
from flair.data import Sentence
from flair.models import SequenceTagger


#TODO exchange to universal dependencies

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

class EnsembleTagger():
    def __init__(self):

        # nltk dependencies
        # for tokenizer
        nltk.download('averaged_perceptron_tagger')
        nltk.download('universal_tagset')


        # stanza dependencies TODO confirm these are UD
        stanza.download('en')
        self.stanza_pipeline = stanza.Pipeline('en',
                                               processors='tokenize,mwt,pos',
                                               tokenize_pretokenized=True)
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
        votes = self.get_all_votes(word_lst)
        tags = self.calculate_votes(votes)
        return tags

    """
        returns dictionary of  majority votes for each token's pos
        dict is of form token_index: majority_tag
        @:param votes list of votes from each tagger 
        @:return dictionary of majority votes
    """
    def calculate_votes(self, votes: [list[list]])-> dict:
        polled_results = {}
        with ThreadPoolExecutor() as executor:
            tagging_tasks = [executor.submit(self.majority_vote, [vote,
                                                index]) for index,
                             vote in enumerate(votes)]
        # Retrieve results from completed tasks
        for task in concurrent.futures.as_completed(tagging_tasks):
            result = task.result()
            polled_results[result[0]] = result[1]
        return polled_results

    """
        this method gets all the votes from the different tokenizers
        for each tagger, we send a list of strings and recieve a list of
        type (token_text, predicted_token_tag)
        @:param spacy_tokens list of strings from spacy's tokenizer
        @:return list of lists returned by each tagger
    """
    def get_all_votes(self, spacy_tokens: list[str])-> list[list[(str, str)]]:
        tagged_tokens_lst = []

        with ThreadPoolExecutor() as executor:
            tagging_tasks = [executor.submit(tagger_func, spacy_tokens) for
                             tagger_func
                             in self.tagger_funcs]

        # retrieve results from completed tasks
        for task in concurrent.futures.as_completed(tagging_tasks):
            result = task.result()
            tagged_tokens_lst.append(result)

        ordered_lst = self.create_lists_from_elements(tagged_tokens_lst)
        return ordered_lst

    """
        this method recieves a list of lists, each one is a list returned 
        by a pos tagger
        it returns one list for each token, which consists of tuples of
        type (token_text, predicted_token_tag)
        @:param list_of_lists list from each pos tagger
        @:return list of one list for each token
    """
    def create_lists_from_elements(self,list_of_lists):
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
    def majority_vote(self, tags_and_token):
        tag_counts = {}

        tags = tags_and_token[0]
        index = tags_and_token[1]
        items = [item[0] for item in tags]
        if not len(set(items)) <= 1:
            raise Exception("For some reason the words are not all the same")
        for vote in tags:
            tag = vote[1]
            if tag in tag_counts:
                tag_counts[tag] += 1
            else:
                tag_counts[tag] = 1
        majority_tag = max(tag_counts, key=tag_counts.get)


        return [index, [tags[0][0], majority_tag]]


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
    def flair_tokenizer(self, token_list:list[str]):
        # since the text is pretokenized, create a sentence object out of it
        sentence = Sentence(token_list)
        self.flair_pipeline.predict(sentence)
        tags = [[token.text, token.labels[0].value] for token in sentence]
        return tags
