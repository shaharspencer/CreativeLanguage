import json
import spacy
import csv
from spacy.symbols import ORTH
from processor import recompile_hyphens
from spacy.tokens import DocBin
import pandas
from docopt import docopt


usage = '''
Processor CLI.
Usage:
    processor.py <csv_path> <make_json>  <json_path> <make_spacy> <spacy_path>
'''


nlp = spacy.load('en_core_web_trf')
# maybe use en_core_web_lg??
special_case = [{ORTH: "i"}, {ORTH: "'m"}]
nlp.tokenizer.add_special_case("i'm", special_case)

# create DocBin object
doc_bin = DocBin(
    attrs=["ORTH", "TAG", "HEAD", "DEP", "ENT_IOB", "ENT_TYPE", "ENT_KB_ID",
           "LEMMA", "MORPH", "POS"])
special_case = [{ORTH: "I"}, {ORTH: "'m"}]
nlp.tokenizer.add_special_case("I'm", special_case)
# recompile hyphens
nlp.tokenizer.infix_finditer = recompile_hyphens()



"""
Creates either one or both of a .spacy doc_bin
and a .json format of given csv file.
    Paremeters:
        csv_path(string): csv file from which to create .spacy file
        create_json(bool): indicates whether to create json file from csv
        json_path(string): json file name to create 
        create_spacy(bool): indicates whether to create spacy obj from csv
        spacy_path(string): spacy file name to create
    Returns: 
        None
"""
def make_json_and_spacy(csv_path, create_json, json_path, create_spacy, spacy_path):
    data = {}
    # open csv
    with open(csv_path, encoding = 'utf-8') as cs:
        csv_reader = csv.DictReader(cs)
        counter = 0

        # iterate over rows
        for rows in csv_reader:
            doc = nlp(rows['text'])
            if create_json:
                sents = [sent.text for sent in doc.sents]
                key = counter
                data[key] = {"sentences": sents}
                rows.pop('text', None)
                rows.pop('', None)
                data[key].update(rows)
            if create_spacy:
                doc_bin.add(doc)
            counter += 1
            if counter == 2:
                break


    if create_json:
        with open(json_path, 'w', encoding='utf-8') as jsonf:
            jsonf.write(json.dumps(data, indent=4))
    if create_spacy:
        doc_bin.to_disk(spacy_path)

if __name__ == '__main__':
    args = docopt(usage)
    csv_path = args['<csv_path>']
    create_json = args['<create_json>']
    json_path = args['<create_json>']
    create_spacy = args['<create_spacy>']
    spacy_path = args['<spacy_path>']
    make_json_and_spacy(csv_path, create_json, json_path, create_spacy, spacy_path)
