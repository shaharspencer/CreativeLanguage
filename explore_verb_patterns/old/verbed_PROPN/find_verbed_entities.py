import spacy
from spacy.tokens import DocBin

from spacy.symbols import ORTH

nlp = spacy.load('en_core_web_trf')


def explore_examples(spacy_path):
    sentence = nlp(
        "Germany and U.S.A are popular countries. I am going to gym tonight. He New-Yorked his way around. I am going to MapQuest it."
        "As she puts it, I tend to treat her like a living doll, often mothering, (even smothering?) her.")
    for token in sentence:
        print(token.text, token.pos_)

    doc_bin = DocBin().from_disk(spacy_path)

    for doc in list(doc_bin.get_docs(nlp.vocab)):
        for sent in doc.sents:
            tmp = nlp(str(sent))
            for ent in sent.ents:
                # produces lots of trash like "weekened"

                if ent.text[-2:] == "ed" or ent.text[-3:] == "ing":
                    print(ent.text, ":", sent)
        for noun_c in doc.noun_chunks:
            if noun_c.text[-2:] == "ed" or noun_c.text[-3:] == "ing":
                print(noun_c.text, ":",noun_c.sent)

if __name__ == '__main__':
    explore_examples(r"/spacy_data/data_from_first_15000_posts.spacy")