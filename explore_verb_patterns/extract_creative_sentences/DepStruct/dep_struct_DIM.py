import pandas as pd
import os


class GetRarestDepStructs:
    def explore_simple_method(self, rarest_dep_structs_csv =
                              r"C:\Users\User\PycharmProjects\CreativeLanguage\extract_creative_sentences\DepStruct\files\no_relcl_freqs.csv",
                              verb_csv =
                              r"C:\Users\User\PycharmProjects\CreativeLanguage\extract_creative_sentences\DepStruct\files\deps_15000_posts_no_relcl.csv",
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
        for colname, colvals in self.verbs_df.iteritems():
            remove_count_from_endstr = len(colname) - len('_COUNT')
            if colname[:remove_count_from_endstr] in self.rarity_set:
                # get all non-zero items in the column
                non_zero_words = colvals[colvals != 0].index
                # get word at indexes
                verbs_with_non_zero_val = self.verbs_df.iloc[non_zero_words]
                # get associated senetences
                for idx, r in verbs_with_non_zero_val.iterrows():
                    sent_path = os.path.join(self.text_file_dir,
                                             r['word'] + colname[
                                                         :remove_count_from_endstr])
                    with open(sent_path, encoding='utf-8') as f:
                        for line in f.readlines():
                            pass





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

