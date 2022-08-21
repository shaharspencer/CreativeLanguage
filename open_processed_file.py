import spacy
from spacy.tokens import DocBin

nlp = spacy.load('en_core_web_trf')
def load_nlp_file():
    doc_bin = DocBin().from_disk("./data.spacy")
    i = 0
    for doc in list(doc_bin.get_docs(nlp.vocab)):
        with open("ent_pos_post/ent_pos_"+str(i), "w") as f:
            for ent in doc:
                f.write(str(ent)+ " " + str(ent.pos_))
                f.write("\n")

        with open("data_vis_post/data_vis_" + str(i) + ".jpg", "w") as f:
            svg = spacy.displacy.render(doc, style='dep', jupyter=False)
            f.write(svg)
        i += 1



if __name__ == '__main__':
    load_nlp_file()