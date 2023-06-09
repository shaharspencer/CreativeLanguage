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


import spacy
import ensemble_tagger

nltk.download('averaged_perceptron_tagger')


nlp = spacy.load("en_core_web_lg")

tagger = ensemble_tagger.EnsembleTagger()


@Language.component("custom_tagger")
def custom_tagger(doc: Doc):
    tags = tagger.get_tags_list(doc)
    sorted_values = [tags[key] for key in sorted(tags.keys())]

    for token, (text, tag) in zip(doc, sorted_values):
        token.tag_ = tag
    return doc






nlp.remove_pipe("tagger")


nlp.add_pipe("custom_tagger", name='tagger', after='tok2vec')

sent = nlp("I few more weeks or months, will tell.")



svg = spacy.displacy.render(sent, style="dep")
output_path = pathlib.Path("h1i.svg")
output_path.open('w', encoding="utf-8").write(svg)


