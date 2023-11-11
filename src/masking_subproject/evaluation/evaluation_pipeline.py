from docopt import docopt

usage = '''
evaluation_pipeline CLI.
number of 
Usage:
    evaluation_pipeline.py
    evaluation_pipeline.py <raw_data_file> <n_sentences>
'''

import src.masking_subproject.format_converters.word_pos_converter as word_pos_converter
import src.masking_subproject.tagging.tag_with_spacy as tag_with_spacy
import spacy_evaluation


def run(raw_data_file=r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\files\raw_data\en_ewt-ud-test.conllu",
        n_sentences=500):
    ud_tags_file = word_pos_converter.run(raw_data_file,
                                          n_sentences=n_sentences)
    spacy_tags_file = tag_with_spacy.run(raw_data_file=raw_data_file,
                                         n_sentences=n_sentences)
    spacy_evaluation.run(gold_standard_file=ud_tags_file,
                         predictions_file=spacy_tags_file)

if __name__ == '__main__':
    args = docopt(usage)
    raw_conllu_file = args["<raw_data_file>"]
    n_sentences = int(args["<n_sentences>"]) if args["<n_sentences>"] != \
                                                "None" else None
    if raw_conllu_file and n_sentences:
        run(raw_data_file=raw_conllu_file, n_sentences=n_sentences)
    else:
        run()