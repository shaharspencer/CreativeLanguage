from conllu import parse
import spacy

# def create_spacy_to_data_tag_mapping():
#     map = {"DATE":
# DATE  --  Absolute or relative dates or periods
# EVENT  --  Named hurricanes, battles, wars, sports events, etc.
# FAC  --  Buildings, airports, highways, bridges, etc.
# GPE  --  Countries, cities, states
# LANGUAGE  --  Any named language
# LAW  --  Named documents made into laws.
# LOC  --  Non-GPE locations, mountain ranges, bodies of water
# MONEY  --  Monetary values, including unit
# NORP  --  Nationalities or religious or political groups
# ORDINAL  --  "first", "second", etc.
# ORG  --  Companies, agencies, institutions, etc.
# PERCENT  --  Percentage, including "%"
# PERSON  --  People, including fictional
# PRODUCT  --  Objects, vehicles, foods, etc. (not services)
# QUANTITY  --  Measurements, as of weight or distance
# TIME  --  Times smaller than a day
# WORK_OF_ART  --  Titles of books, songs, etc.
# }
def get_spacy_ner():
    """ see which ner tags spacy has """
    nlp = spacy.load("en_core_web_lg")
    for label in nlp.get_pipe("ner").labels:
        print(label + ", ")

def open_conllu(raw_data_file):
    with open(raw_data_file, 'r', encoding='utf-8') as conllu_file:
        conllu_content = parse(conllu_file.read())
    return conllu_content

if __name__ == '__main__':
    open_conllu(r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\NER\raw_data\en_ewt-ud-test.conllu")
    get_spacy_ner()
