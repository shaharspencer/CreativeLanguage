import spacy
from conllu import parse

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to convert CoNLL-U format to tagged tokens
def convert_conllu_to_tagged_text(conllu_content, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in conllu_content:
            sentence_text = " ".join([token_info['form'] for token_info in sentence])
            doc = nlp(sentence_text)

            for token in doc:
                f.write(f"{token.text} {token.pos_}\n")

            f.write("\n")

# Replace 'your_input_file.conllu' and 'output_tagged_text.txt' with your file paths


if __name__ == '__main__':
    args = docopt(usage)
    file_to_process = args["<file_to_proccess>"]
    n_sentences = int(args["<n_sentences>"])
    input_file = 'your_input_file.conllu'
    output_file = 'output_tagged_text.txt'

    with open(input_file, 'r', encoding='utf-8') as conllu_file:
        conllu_content = parse(conllu_file.read())

    convert_conllu_to_tagged_text(conllu_content, output_file)

    print(f'Data converted and saved to {output_file} with spaCy POS tags')
    pass