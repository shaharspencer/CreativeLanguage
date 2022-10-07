import spacy
import pandas
from spacy.tokens import DocBin
from docopt import docopt
from pathlib import Path
from spacy.symbols import ORTH
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex

usage = '''
Processor CLI.
Usage:
    processor.py <file_to_process>
    processor.py <file_to_process> <tag_parts_of_speech> <create_dependency_map>
'''

"""
Creates a .spacy doc_bin of given csv file.

    Paremeters:
        file(string): file from which to create .spacy file
        print_pos(bool): indicates whether to create file of parts of speech
        view_dep(bool): indicates whether to create dependency tree file
    Returns: 
        None
"""
def process_file_and_create_nlp_objs(file, print_pos=False, view_dep=False, no_split_hyphens=True):
    # load model
    nlp = spacy.load('en_core_web_trf')
    special_case = [{ORTH: "i"},{ORTH: "'m"}]
    nlp.tokenizer.add_special_case("i'm", special_case)


    # create DocBin object
    doc_bin = DocBin(attrs=["ORTH", "TAG", "HEAD", "DEP", "ENT_IOB", "ENT_TYPE", "ENT_KB_ID", "LEMMA", "MORPH", "POS"])
    special_case = [{ORTH: "I"}, {ORTH: "'m"}]
    nlp.tokenizer.add_special_case("I'm", special_case)
    if no_split_hyphens:
        nlp.tokenizer.infix_finditer = recompile_hyphens

    # read csv file
    df = pandas.read_csv(file)
    i = 0
    # iterate over rows in csv and create nlp object from text
    for index, row in df.iterrows():
        print(index)

        doc = nlp(row['text'])
        if print_pos:
            with open("ent_pos_pre/ent_pos_" + str(i), "w") as f:
                for ent in doc:
                    f.write(str(ent) + " " + str(ent.pos_))
                    f.write("\n")

        if view_dep:
            for sent in doc.sents:
                svg = spacy.displacy.render(sent, style='dep', jupyter=False)
                output_path = Path(
                    "data_vis_pre/data_vis_" + str(i)+".svg")
                output_path.open("w", encoding="utf-8").write(svg)
                break
        doc.retokenize()
        doc_bin.add(doc)
        i += 1
        if i == 20:
            break

    # push DocBin object to disk
    doc_bin.to_disk("./data_from_first_20_posts.spacy")


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


if __name__ == '__main__':
    args = docopt(usage)
    if args['<file_to_process>']:
        source_file = args['<file_to_process>']
        print_pos, dep = False, False
        # if user wants to tag parts of speech
        if args['<tag_parts_of_speech>']:
            print_pos = True
        # if user wants to create dependency map
        if args['<create_dependency_map>']:
            dep = True
        process_file_and_create_nlp_objs(source_file, print_pos, dep)
