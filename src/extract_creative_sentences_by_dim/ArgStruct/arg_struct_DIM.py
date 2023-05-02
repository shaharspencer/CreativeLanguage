import csv

import pandas as pd
import os

from docopt import docopt

import src.utils.path_configurations as paths
import spacy

usage = '''
arg_struct_DIM CLI.
Usage:
    arg_struct_DIM.py <frequency_csv> <sentence_source_file> <num_of_posts>
'''


class GetRarestArgStructs:
    def __init__(self, verb_csv, sents_csv, model = "en_core_web_lg"):
        self.verb_csv = os.path.join(paths.files_directory,
                                             paths.dependency_set_directory,
                                             paths.dependency_set_source_files,
                                             verb_csv)
        self.sents_csv = os.path.join(paths.files_directory,
                                     paths.dependency_set_directory,
                                     paths.dependency_set_source_files,
                                     sents_csv)
        self.nlp = spacy.load(model)


    def explore_simple_method(self, dep_set_frequency,
                              verb_csv, num_of_posts,
                              k_val = 20):
        dep_set_frequency_src = os.path.join(paths.files_directory,
                                   paths.dependency_set_directory,
                                   paths.dependency_set_source_files,
                                   dep_set_frequency)
        verbs_csv_src = os.path.join(paths.files_directory,
                                   paths.dependency_set_directory,
                                   paths.dependency_set_source_files,
                                   verb_csv)


        self.rarity_df = pd.read_csv(dep_set_frequency_src).sort_values(
            by='count', ascending=True).copy()
        from datetime import datetime
        datetime = datetime.today().strftime('%Y_%m_%d')
        self.output_path = "dependency_set_sents_by_count_" + datetime + ".csv"
        self.output_path = os.path.join(paths.files_directory,
                                        paths.rare_sents_directory,
                                        self.output_path)


        # get dep structures by rarity
        self.verbs_df = pd.read_csv(verbs_csv_src)

        self.rarity_set = self.get_k_rarest_sents_from_rare_structs(k_val)

        self.get_lowest_sentences()
    """
    get creative setences by entropy measures.
    we want sentences that:
    1. appear in at least 2 different complement sets
    we want to:
    1. measure the entropy of each verb across the different complement sets
    2. for the verbs with the lowest entropy, take the least 
    common complement set
    3. print verbs' sentences with the least common complement set
    """

    def explore_by_entropy(self, output_path, k:int):
        # open counter csv
        verb_df = pd.read_csv(self.verb_csv)
        # open sentences df
        sents_df = pd.read_csv(self.sents_csv)
        # screen out verbs with low set count
        verb_df = verb_df[verb_df["num_sets"] > 1]
        sorted_dataframe = verb_df.sort_values(["entropy"],
                                          ascending=True)
        # open output csv
        output_csv = open(output_path, "w", encoding='utf-8', newline='')

        # define csv columns
        fields = ["lemma", "verb form",
                            "Sentence",
                           "dep set", "count of dep set",
                           "perc of dep set", "entropy",
                           "index of verb",
                           "Doc index",
                            "Sent index"]

        writer = csv.DictWriter(f=output_csv, fieldnames=fields)

        d = print_fieldnames(fields)
        writer.writerow(d)
        # iterate over top rows in dataframe, up until we get to k sentences
        counter = 0

        for index, row in sorted_dataframe.iterrows():
            if counter == 150:
                break
            counter += 1
            # get rarest dependency set for current lemma
            rarest_dep_struct, min_count = None, 0
            for i_col, col in row.items():
                if i_col.endswith("_COUNT") and \
                    (not rarest_dep_struct or col < min_count) and col > 0:
                    rarest_dep_struct = i_col[:len(i_col)-len("_COUNT")]
                    min_count = col

            # get subset of dataframe that represents current lemma
            # with rarest dep struct
            current_lemma_df = sents_df[(sents_df["Lemma (V)"] ==
                                        row["Lemma (V)"]) &
                                        (sents_df["Dep struct"] == rarest_dep_struct)]
            for s_i, s_row in current_lemma_df.iterrows():
                if rarest_dep_struct == "NO_DEPS":
                    dep_set = set()
                else:
                    dep_set = set(rarest_dep_struct.split("_"))

                index_of_verb = self.get_index_of_verb(lemma=row["Lemma (V)"],
                                                       verb_form
                                                       =s_row["Verb form"],
                                                       dep_set =
                                                       dep_set,
                                                       sent=s_row["Sentence"])
                n_dict = {"lemma": row["Lemma (V)"],
                          "verb form": s_row["Verb form"],
                          "Sentence": s_row["Sentence"],
                          "dep set": rarest_dep_struct,
                          "count of dep set": row[rarest_dep_struct + "_COUNT"],
                          "perc of dep set": row[rarest_dep_struct + "%"],
                          "entropy": row["entropy"],
                           "Doc index": s_row["Doc index"],
                           "Sent index": s_row["Sent index"],
                          "index of verb": index_of_verb
                          }

                writer.writerow(n_dict)

    # todo idk what this function is
    def get_lowest_sentences(self,):
        # new_df = self.verbs_df[self.verbs_df.loc[self.verbs_df['Dep struct'].isin(self.rarity_set)]]
        new_df = self.verbs_df.loc[self.verbs_df['Dep struct'].isin(self.rarity_set)].copy()
        percent_column = []
        count_column = []

        for indx, row in new_df.iterrows():
            dep_row = self.rarity_df[self.rarity_df['dep_struct'] == row["Dep struct"]].squeeze()
            percent_column.append(dep_row["%of_total"])
            count_column.append(dep_row["count"])
        new_df["count of dep struct"] = count_column
        new_df["percent of dep struct"] = percent_column
        new_df.to_csv(self.output_path, index = False)

    def get_k_rarest_sents_from_rare_structs(self, k)-> set:
        rarity_set_count = 0
        rarity_set = set()
        for index, row in self.rarity_df.iterrows():
            rarity_set.add(row['dep_struct'])
            rarity_set_count += row['count']
            if rarity_set_count > k:
                return rarity_set
    """
    given parameters describing the verb, we find the index spacy gives 
    to the verb within the sentence.
    """
    def get_index_of_verb(self, lemma:str, verb_form:str, dep_set:set,
                          sent: str)->int:
        doc = self.nlp(sent)
        for token in doc:
            token_deps = list([
                    child for child in token.children
                if child.dep_ != "punct" and child.dep_ != "SPACE" and child.dep_ != "dep"])
            token_form = token.text
            token_lemma = token.lemma_
            if token_lemma == "contain":
                i=1

            if "_" in verb_form:
                child = self.get_child_with_specific_dependency(list(token_deps),
                                                                            "prt")
                if not child:
                    continue
                token_form += "_" + child.text
                token_lemma += "_" + child.text
                token_deps.remove(child)
            token_deps = set([child.dep_ for child in token_deps])

            if token_form == verb_form and token_lemma == lemma \
                    and token_deps == dep_set:
                return token.i
        return -1

    def get_child_with_specific_dependency(self, dep_list: list[spacy.tokens],
                                           dep: str):
        for child in dep_list:
            if child.dep_ == dep:
                return child
        return ""


def print_fieldnames(given_lst: iter):
    dic = {}
    for fieldname in given_lst:
        dic[fieldname] = fieldname
    return dic

if __name__ == '__main__':
    #TODO fix source files
    args = docopt(usage)
    from datetime import datetime
    r = GetRarestArgStructs(args["<frequency_csv>"],
                            args["<sentence_source_file>"])
    datetime = datetime.today().strftime('%Y_%m_%d')
    # r.explore_simple_method(args["<frequency_csv>"],
    #                         args["<sentence_source_file>"],
    #                         num_of_posts=args["<num_of_posts>"], k_val=20)
    entropy_output_path = "rarest_sents_by_entropy_" + datetime + ".csv"
    r.explore_by_entropy(output_path=entropy_output_path, k = 100)

