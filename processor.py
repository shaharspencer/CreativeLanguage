import spacy
import pandas
from spacy.tokens import DocBin
from docopt import docopt


usage = '''
Processor CLI.
Usage:
    processor.py <file_to_process>
    processor.py <file_to_process> <tag_parts_of_speech> <create_dependency_map>
'''


def process_file_and_create_nlp_objs(file, print_pos=False, view_dep=False):
    # load model
    nlp = spacy.load('en_core_web_trf')

    # create DocBin object
    doc_bin = DocBin(attrs=["POS", "LEMMA", "ENT_IOB", "ENT_TYPE"])

    # read csv file
    df = pandas.read_csv(file)
    i = 0
    # iterate over rows in csv and create nlp object from text
    for index, row in df.iterrows():
        doc = nlp(row['text'])
        if print_pos:
            with open("ent_pos_pre/ent_pos_" + str(i), "w") as f:
                for ent in doc:
                    f.write(str(ent) + " " + str(ent.pos_))
                    f.write("\n")

        if view_dep:
            with open("data_vis_pre/data_vis_"+str(i)+".svg", "w") as f:
                svg = spacy.displacy.render(doc, style='dep', jupyter=False)
                f.write(svg)

        doc_bin.add(doc)

        i += 1
        if i == 5:
            break
    # push DocBin object to disk
    doc_bin.to_disk("./data.spacy")


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
