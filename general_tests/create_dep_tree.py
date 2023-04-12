import spacy
import pathlib
import os
from spacy import displacy
import pandas as pd


class Renderer:
    def __init__(self, model_to_use = "en_core_web_lg"):
        self.nlp = self.initialize_nlp(model_to_use)

    def initialize_nlp(self, model):
        return spacy.load(model)
    #     nlp.tokenizer.infix_finditer = self.recompile_hyphens()
    #     special_case = [{ORTH: "I"}, {ORTH: "'m"}]
    #     nlp.tokenizer.add_special_case("I'm", special_case)
    #
    #     return nlp
    #
    # def recompile_hyphens(self):
    #     infixes = (
    #                       LIST_ELLIPSES
    #                       + LIST_ICONS
    #                       + [
    #                           r"(?<=[0-9])[+\-\*^](?=[0-9-])",
    #                           r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
    #                               al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
    #                           ),
    #                           r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
    #
    #                   r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
    #     ]
    #     )
    #
    #     infix_re = compile_infix_regex(infixes)
    #
    #     return infix_re.finditer


    def output_sent_to_svg(self, sent: str, output_path: str):

        sent = self.nlp(sent)

        svg = spacy.displacy.render(sent, style="dep")
        output_path = pathlib.Path(output_path)
        output_path.open('w', encoding="utf-8").write(svg)

    """
    takes a csv file with a column names "Sentence" and outputs a rendering
    for each sentence in the file
    """
    def create_renderings_with_csv_file(self, csv_path, output_dir):
        sents_df = pd.read_csv(csv_path, encoding='utf-8')
        for ind, r in sents_df.iterrows():
            self.output_sent_to_svg(r["Sentence"],
                                    os.path.join(output_dir, str(ind) + ".svg"))


if __name__ == '__main__':
    # sent = "In het kader van kernfusie op aarde:  MAAK JE EIGEN WATERSTOFBOM   How to build an H-Bomb From: ascott@tartarus.uwa.edu.au (Andrew Scott) Newsgroups: rec.humor Subject: How To Build An H-Bomb (humorous!)"

    sent = "First transform the gas into a liquid by subjecting it to pressure."
    renderer = Renderer(model_to_use="en_core_web_lg")
    renderer.output_sent_to_svg(sent, "sent.svg")




