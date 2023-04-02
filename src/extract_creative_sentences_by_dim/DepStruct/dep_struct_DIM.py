import pandas as pd
import os


class GetRarestDepStructs:
    def explore_simple_method(self, rarest_dep_structs_csv =
                              r"C:\Users\User\PycharmProjects\CreativeLanguage\explore_verb_patterns\extract_creative_sentences\DepStruct\files\dep_freqs_15000_sents.csv",
                              verb_csv =
                              r"C:\Users\User\PycharmProjects\CreativeLanguage\explore_verb_patterns\extract_creative_sentences\DepStruct\files\first_15000_posts_sents_deg_struct_dim.csv",
                              text_file_dir='None',
                              k_val = 20):
        self.rarity_df = pd.read_csv(rarest_dep_structs_csv).sort_values(
            by='count', ascending=True)
        self.text_file_dir = text_file_dir
        # get dep structures by rarity
        self.verbs_df = pd.read_csv(verb_csv)

        self.rarity_set = self.get_k_rarest_sents_from_rare_structs(k_val)

        self.get_lowest_sentences()



    def get_lowest_sentences(self,):
        new_df = self.verbs_df.loc[
            self.verbs_df['Dep struct'].isin(self.rarity_set)].copy()

        percent_column = []
        count_column = []

        for indx, row in new_df.iterrows():
            dep_row = self.rarity_df[self.rarity_df['dep_struct'] == row["Dep struct"]].squeeze()
            percent_column.append(dep_row["%of_total"])
            count_column.append(dep_row["count"])
        new_df["count of dep struct"] = count_column
        new_df["percent of dep struct"] = percent_column
        from datetime import datetime
        datetime = datetime.today().strftime('%Y_%m_%d')
        self.output_path = "dependent_order_by_count_" + datetime + ".csv"
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
    r = GetRarestDepStructs()
    r.explore_simple_method()

