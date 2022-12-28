import spacy
import pandas
from spacy.tokens import DocBin
from docopt import docopt
from pathlib import Path
from spacy.symbols import ORTH
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex






nlp = spacy.load("en_core_web_trf")





def recompile_hyphens():
    infixes = (
                      LIST_ELLIPSES
                      + LIST_ICONS
                      + [
                          r"(?<=[0-9])[+\-\*^](?=[0-9-])",
                          r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                              al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
                          ),
                          r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),

                  r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
    ]
    )

    infix_re = compile_infix_regex(infixes)


    return infix_re.finditer
nlp.tokenizer.infix_finditer = recompile_hyphens()
import pathlib
import os
from spacy import displacy


special_case = [{ORTH: "I"}, {ORTH: "'m"}]
nlp.tokenizer.add_special_case("I'm", special_case)



sent = nlp("What should have been an hour trip took us four hours.")

svg = spacy.displacy.render(sent, style="dep")
output_path = pathlib.Path(os.path.join("./", "sentence.svg"))
output_path.open('w', encoding="utf-8").write(svg)

