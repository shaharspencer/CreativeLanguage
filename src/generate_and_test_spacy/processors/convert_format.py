"""
    this file converts spacy object to conllu format
"""

from spacy.tokens import DocBin


usage = '''
convert_format CLI.
Usage:
    convert_format.py <file_to_process> <number_of_blogposts>
'''

def convert_docbin_to_conllu(docbin, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for doc in docbin.get_docs():
            conllu_lines = []
            for sent in doc.sents:
                conllu_lines.extend(convert_sent_to_conllu(sent))
            f.write("\n".join(conllu_lines))
            f.write("\n\n")

def convert_sent_to_conllu(sent):
    conllu_lines = []
    for i, token in enumerate(sent):
        line_parts = [
            str(i + 1),          # ID
            token.text,          # FORM
            token.lemma_,        # LEMMA
            token.pos_,          # UPOS
            token.tag_,          # XPOS
            "_",                 # FEATS
            str(token.head.i + 1) if token.head is not token else "0",  # HEAD
            token.dep_,          # DEPREL
            "_",                 # DEPS
            "_",                 # MISC
        ]
        conllu_lines.append("\t".join(line_parts))
    return conllu_lines

# Example usage
docbin = DocBin().from_disk("your_input_docbin.spacy")
convert_docbin_to_conllu(docbin, "output.conllu")