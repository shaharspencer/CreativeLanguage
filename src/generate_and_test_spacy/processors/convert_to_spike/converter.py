import pandas as pd
import spacy
import zipfile

from docopt import docopt
from spacy.tokens import DocBin

usage = '''
converter CLI.
Usage:
    converter.py <blogcorpus>
'''

NEW_DOCUMENT = ">>>DOCUMENT_SEPARATOR<<<\n"

class SpikeConverter:
    def __init__(self, data_file):
# Load the spaCy model
        self.nlp = spacy.load('en_core_web_lg')
        # self.spacy_file = spacy_file

# Load the DocBin object
        self.dataframe = pd.read_csv(blogcorpus, encoding='utf-8').head(20)

# Create a list to store the document contents
    def convert(self, JUMPS = 10000):
        documents = []

        for i in range(0, len(self.dataframe), JUMPS):
            document = ""
            for j in range(JUMPS):
                if i + j >= len(self.dataframe):
                    break
                document_title = f"@title: Blogpost number {i+j}"
                # check if we have exceeded the blogpost length

                document_text = self.dataframe.iloc[i + j]["text"]
                document += f'{document_title}\n{document_text}\n'
                if j < JUMPS - 1:
                    document += NEW_DOCUMENT
                print(f"processed blogpost number {i+j}\n")
            documents.append(document)



        # Create a ZIP file to store the documents
        with zipfile.ZipFile('spike_dataset_40000.zip', 'w') as zipf:
            # Add each document as a separate file to the ZIP
            for i, document in enumerate(documents):
                filename = f'document_{i + 1}.txt'
                zipf.writestr(filename, document)

        print('Conversion completed. The SPIKE dataset is stored in spike_dataset.zip.')



if __name__ == '__main__':
    args = docopt(usage)
    blogcorpus = args["<blogcorpus>"]

    obj = SpikeConverter(blogcorpus)
    obj.convert()