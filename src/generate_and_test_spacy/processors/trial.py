import pathlib

import nltk
import pandas as pd
import spacy
import pandas
from spacy import Language
from spacy.tokens import Doc
from spacy.tokens import DocBin
from docopt import docopt
# from spacy.symbols import ORTH
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
import os
from tqdm import tqdm
import stanza
from flair.data import Sentence
from flair.models import SequenceTagger


import spacy
import ensemble_tagger

nltk.download('averaged_perceptron_tagger')


nlp = spacy.load("en_core_web_lg")

# tagger = ensemble_tagger.EnsembleTagger()


# nlp.remove_pipe("tagger")

@Language.component("custom_tagger")
def nltk_tagger(doc):
    #nltk
    token_lst = [token.text for token in doc]
    tagged_tokens_nltk = nltk.pos_tag(token_lst)
    # stanza_nlp = stanza.Pipeline('en', processors='tokenize,pos', tokenize_pretokenized=True)
    # #stanza
    # taggings = stanza_nlp([token_lst])
    # tags = [word for word in taggings.sentences[0].words]
    # #flair
    flair_pipeline = SequenceTagger.load("flair/upos-english")
    sentence = Sentence(token_lst)
    flair_pipeline.predict(sentence)
    tags = [[token.text, token.tag] for token in sentence]
    # return tags

    # # use nltk
    for token, (text, tag) in zip(doc, tags):
        token.pos_ = "NOUN"
    #
    return doc


# @Language.component("custom_tagger")
# def nltk_tagger(doc):
#     for token in doc:
#         print(
#             f"token text is {token.text}, token lemma is {token.lemma_}, token pos is {token.pos_}")
#
#     token_lst = [token.text for token in doc]
#     tags = tagger.get_tags_list(token_lst)
#     tags = [tags[key] for key in sorted(tags.keys())]
#
#     for token, (text, tag) in zip(doc, tags):
#         token.pos_ = tag
#     return doc

nlp.add_pipe("custom_tagger", last=True)

sent = nlp("I few more weeks or months, will tell.")

for token in sent:
    print(f"token text is {token.text}, token lemma is {token.lemma_}, token pos is {token.pos_}")

svg = spacy.displacy.render(sent, style="dep")
output_path = pathlib.Path("h1i.svg")
output_path.open('w', encoding="utf-8").write(svg)


