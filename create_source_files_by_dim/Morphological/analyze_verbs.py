from spacy.tokens import DocBin
import csv
import os
import spacy
from spacy.tokens import DocBin
from zipfile import ZipFile



#TODO: are these the correct parts of speech?
# nlp.get_pipe showed other parts of speech...

#TODO: general debug, see if there are issues

#TODO: output files with 50000 posts

parts_of_speech = ["VERB", "PROPN", "PART", "NUM", "X", "PUNCT",
                        "ADJ",
                        "ADP", "ADV",
                        "AUX", "CCONJ", "DET", "INTJ", "NOUN", "PRON",
                        "SCONJ", "CONJ",
                        "SYM"]




"""
For every word that is used as a verb on some occasion, 
Count the frequency of the word.lemma_ in different pos
And save occurences of that word in the part of speech

    Paremeters:
        spacy_path(string): a path to a .spacy file from which to create the csv
    Returns:
        for_csv(dict): 
         for every word that is a verb, for the lemma of the word,
            for every pos:
                - save sentences in which it is used as that part of speech
                - count occurences
"""

class AnalyzeVerbs:
    def __init__(self, model="en_core_web_lg",
                 spacy_directory=
                 r"C:\Users\User\OneDrive\Documents\CreativeLanguageOutputFiles\training_data\spacy_data\withought_context_lg_model",
                 spacy_file_path = r"data_from_first_1_lg_model.spacy"):
        self.nlp = spacy.load(model)

        # self.spacy_part_of_speech = self.nlp.get_pipe("tagger").labels
        self.spacy_part_of_speech = parts_of_speech

        self.spacy_path = os.path.join(spacy_directory, spacy_file_path)
        self.doc_bin = DocBin().from_disk(self.spacy_path)
        # find all words which at some point are classified as verbs
        self.words_classed_as_verb = self.find_all_verbs_in_file()
        #initialize dictionary
        self.verb_dict = {}
        #fill dictionary
        self.analyze_verbs()

    """
    creates a set of all verbs that were at some point in the files
    classified as verbs
    @:param by_lemma: save the words by their lemma
    """
    def find_all_verbs_in_file(self)->set:
        verbs = set()
        for doc in self.doc_bin.get_docs(self.nlp.vocab):
            for token in doc:
                if not self.verify_we_want_to_add_token(token):
                    continue
                verbs.add(token.lemma_.lower())
        return verbs

    """
    create dictionary with all words that are at some point classified as verbs
    we know they were clasified as verbs using the set words_classed_as_verbs
    save all instances in different parts of speech, lemmas and more info
    """
    def analyze_verbs(self):
        for sentence in self.doc_bin.get_docs(self.nlp.vocab):
            for token in sentence:
                if (token.lemma_.lower() in self.words_classed_as_verb):
                    self.add_token_to_dict(token)


    """
    adds current instance of word to dictionary
    adds template for word if not already in the dictionary
    """

    def add_token_to_dict(self, token):
        if token.text == "have":
            x = 0
        self.add_token_template_to_dict(token)
        # convert all letters but first letter to lowercase
        self.verb_dict[token.lemma_.lower()][token.pos_]["Instances"].add(
            (token.text, token.sent.text, token.doc.user_data["DOC_INDEX"],
             token.doc.user_data["SENT_INDEX"]))
        self.verb_dict[token.lemma_.lower()][token.pos_]["lemma"] = \
            token.lemma_
        self.verb_dict[token.lemma_.lower()][token.pos_]["Counter"] += 1

    """
    if token is not in the dictionary add it to the 
    dictionary with a pos template
    """
    def add_token_template_to_dict(self, token:spacy.tokens):
        if not token.lemma_.lower() in self.verb_dict.keys():
            self.verb_dict[token.lemma_.lower()] = self.dict_template()


    """
    create csv with all differnt count of parts of speech
    """
    def write_dict_to_csv(self, output_file_directory =
    r"C:\Users\User\OneDrive\Documents\CreativeLanguageOutputFiles\morphological_dimension\source_files\pos_distribution_csv",
                          output_file_name = r"first_50000_lg_model.csv"):
        output_path = os.path.join(output_file_directory, output_file_name)
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ["word"]
            fields, pos_dict = self.csv_pos_template()

            fieldnames += fields

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(pos_dict)

            for word in self.verb_dict.keys():
                n_dict = {'word': word}
                total = sum(
                    [self.verb_dict[word][pos]["Counter"] for pos in self.spacy_part_of_speech])
                for pos in self.spacy_part_of_speech:
                    n_dict[pos + "_count"] = self.verb_dict[word][pos]["Counter"]
                    n_dict[pos + "%"] = self.verb_dict[word][pos]["Counter"] / total
                    n_dict[pos + "_lemma"] = self.verb_dict[word][pos]["lemma"]

                writer.writerow(n_dict)

    def create_text_files(self, folder_directory=
    r"C:\Users\User\OneDrive\Documents\CreativeLanguageOutputFiles\morphological_dimension\source_files\pos_distribution_sentences",
                          folder_name = r"first_50000_posts_with_lg"):
        zip_file_dir = os.path.join(folder_directory, folder_name + ".zip")
        with ZipFile(zip_file_dir, 'w') as file_dir:
            for word in self.verb_dict.keys():
                for pos in self.spacy_part_of_speech:
                    file = self.create_csv_for_pos(word, pos)
                    if file != "":
                        # add to zip folder
                        file_dir.write(file)
                        # remove from current folder
                        os.remove(file)


    def create_csv_for_pos(self, word, pos)->str:
        if not (self.verb_dict[word][pos]["Instances"]):
            return ""
        if not self.verify_word(word):
            return ""
        p = word + "_" + pos + ".csv"
        with open(p,
                  mode="w", encoding='utf-8', newline="") as f:
            fields = ["lemma", "word form", "sentence",
                      "doc index",
                      "sent index"]
            d = self.fieldnames_for_csv_by_pos(fields)
            writer = csv.DictWriter(f=f, fieldnames=fields)
            writer.writerow(d)

            for sent in self.verb_dict[word][pos]["Instances"]:
                n_dict = self.create_instance_row(word, sent)
                writer.writerow(n_dict)
        return p

    def create_instance_row(self, word, sent)->dict:
        verb_form, sentence, doc_index, sent_index = sent[
                                                         0], \
                                                     sent[
                                                         1], \
                                                     sent[
                                                         2], \
                                                     sent[
                                                         3]
        n_dict = {"lemma": word,
                  "word form": verb_form,
                  "sentence": sentence,
                  "doc index": doc_index,
                  "sent index": sent_index}
        return n_dict

    def verify_word(self, word: str)->bool:
        illegal = [':', '*', "?", "<", ">", "|", '"', chr(92), chr(47)]
        return not any(ill in word for ill in illegal)

    """
    check that this token is a verb
    """
    def verify_we_want_to_add_token(self, token:spacy.tokens)->bool:
        return token.pos_ == "VERB"

    """
    create template for dictionary for analyze_verbs function
        Parameters: None
        return: dict(dict) : dictionary for every word
        containe
    """
    def dict_template(self):
        dict = {}
        for pos in self.spacy_part_of_speech:
            dict[pos] = {"lemma": "", "Counter":0, "Instances":set()}
        return dict

    """
    create template for csv shopwing the distribution of tha parts of speech
    """

    def csv_pos_template(self):
        pos_template = []
        for pos in self.spacy_part_of_speech:
            pos_template.append(pos + "_lemma")
            pos_template.append(pos + "_count")
            pos_template.append(pos + "%")
        d = {"word": "word"}
        for pos in self.spacy_part_of_speech:
            d[pos + "_lemma"] = pos + "_lemma"
            d[pos + "_count"] = str(pos) + "_count"
            d[pos + "%"] = pos + "%"

        return pos_template, d

    """
    get fieldnames for csv that represents all sentences with word in a 
    specific pos
    """
    def fieldnames_for_csv_by_pos(self, given_lst: iter):
        dic = {}
        for fieldname in given_lst:
            dic[fieldname] = fieldname
        return dic





if __name__ == '__main__':
    verb_anazlyzer = AnalyzeVerbs()

    verb_anazlyzer.write_dict_to_csv()

    verb_anazlyzer.create_text_files()
