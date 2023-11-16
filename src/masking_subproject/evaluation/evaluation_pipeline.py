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
import src.masking_subproject.tagging.tag_with_mask as tag_with_mask
from src.masking_subproject.evaluation import masking_evaluation
from src.masking_subproject.evaluation import spacy_evaluation


#
def run(raw_data_file=r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\files\raw_data\en_ewt-ud-test.conllu",
        n_sentences=50000):
    ud_tags_file = word_pos_converter.run(raw_data_file=raw_data_file,
                                          n_sentences=n_sentences)
    spacy_tags_file = tag_with_spacy.run(raw_data_file=raw_data_file,
                                         n_sentences=n_sentences)
    combined_tags_file = spacy_evaluation.run(gold_standard_file=ud_tags_file,
                         predictions_file=spacy_tags_file, n_sentences=n_sentences)
    mask_combined = tag_with_mask.run(combined_dataframe=combined_tags_file, n_sentences=n_sentences)
    mask_evaluation = masking_evaluation.run(data_file_path=mask_combined)



if __name__ == '__main__':
    args = docopt(usage)
    raw_conllu_file = args["<raw_data_file>"]
    n_sentences = int(args["<n_sentences>"]) if args["<n_sentences>"] != \
                                                "None" else None
    if raw_conllu_file and n_sentences:
        run(raw_data_file=raw_conllu_file, n_sentences=n_sentences)
    else:
        run()