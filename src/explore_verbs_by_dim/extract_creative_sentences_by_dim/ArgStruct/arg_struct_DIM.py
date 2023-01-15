import pandas as pd
import os

from docopt import docopt

import utils.path_configurations as paths

usage = '''
arg_struct_DIM CLI.
Usage:
    arg_struct_DIM.py <frequency_csv> <sentence_source_file> <num_of_posts>
'''


class GetRarestArgStructs:
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




        # get sentences with rarest dep structs

if __name__ == '__main__':
    #TODO fix source files
    args = docopt(usage)
    from datetime import datetime
    r = GetRarestArgStructs()
    datetime = datetime.today().strftime('%Y_%m_%d')
    r.explore_simple_method(args["<frequency_csv>"],
                            args["<sentence_source_file>"],
                            num_of_posts=args["<num_of_posts>"], k_val=20)

