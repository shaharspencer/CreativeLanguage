# https://www.youtube.com/watch?v=iTBwgW-i1zU&t=80s

import spacy
import pandas
from spacy.tokens import DocBin
from docopt import docopt
from pathlib import Path
from tqdm import tqdm
nlp = spacy.load('en_core_web_trf')


class CreativeVerbs():
    def __init__(self, sent):
        self.nlp_sent = nlp(sent)
        self.potential_verbs = [verb for verb in self.nlp_sent if verb.pos_ == "VERB"]
        self.funcs = [self.NounVerb,
                      self.SymbolVerb,
                      self.AdjectiveVerb,
                      self.AllusionVerb,
                      self.SimileVerb]
        # perhaps parse over data and recognize target verbs? perhaps save with index?

    def NounVerb(self):
        # case 1:
        # big letter at beggining of verb
        pass


    def SymbolVerb(self):
        pass
        # 1. pattern: sentences of the type x = y, eg: Shahar = Programmer
        # if spacy identifies the sign = as a verb, we probably have this pattern
        if "=" in self.potential_verbs:
            print("Found creative pattern: type x = y, eg: Shahar = Programmer")

    def AdjectiveVerb(self):
        pass

    def AllusionVerb(self):
        pass

    def SimileVerb(self):
        pass


if __name__ == '__main__':
    # read object with sentences
    sent = "food = good; or at least its the most similar word you can form with a standard 'qwerty' keyboard."
