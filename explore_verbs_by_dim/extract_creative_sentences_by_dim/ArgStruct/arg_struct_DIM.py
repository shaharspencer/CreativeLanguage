import pandas as pd
import os


class GetRarestArgStructs:
    def explore_simple_method(self, rarest_dep_structs_csv =
                              r"C:\Users\User\PycharmProjects\CreativeLanguage\explore_verb_patterns\extract_creative_sentences\ArgStruct\files\dep_freqs_15000_sents.csv",
                              verb_csv =
                              r"C:\Users\User\PycharmProjects\CreativeLanguage\explore_verb_patterns\extract_creative_sentences\ArgStruct\files\first_15000_posts_sents_arg_struct_dim.csv",
                              k_val = 20):
        from datetime import datetime
        datetime = datetime.today().strftime('%Y_%m_%d')
        self.output_path = "dependency_set_by_count_" + datetime + ".csv"


        self.rarity_df = pd.read_csv(rarest_dep_structs_csv).sort_values(
            by='count', ascending=True).copy()
        # get dep structures by rarity
        self.verbs_df = pd.read_csv(verb_csv)

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

        new_df.to_csv(self.output_path)




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
    r = GetRarestArgStructs()
    r.explore_simple_method()

