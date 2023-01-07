import os.path

import docopt
import pandas as pd

usage = '''
Processor CLI.
Usage:
    ExploreVerbDependencies.py
    ExploreVerbDependencies.py <csv_path>
    ExploreVerbDependencies.py <csv_path> <perc> <text_file_source_dir>
'''

class OutPutRowwise:
    def __init__(self, csv_path = r"C:\Users\User\PycharmProjects\CreativeLanguage\explore_verb_patterns\Dependencies\deps_15000_posts_no_relcl.csv"
                 ):
        self.csv_path = csv_path
        # self.output_path = output_path

    def Output(self, percent = 0.01, dir = "suprising_sents_per_word"):
        df = pd.read_csv(self.csv_path)
        rare_per_verb_dict = {}

        for rowName, row_vals in df.iterrows():
            key = row_vals[0]
            for ColName, colVal in row_vals.iteritems():
                # if for this verb, this is a rare dependency
                if ColName.endswith("%") and colVal < percent:
                    if key in rare_per_verb_dict.keys():
                        rare_per_verb_dict[key][ColName[:-1]] = colVal
                    else:
                        rare_per_verb_dict[key] = {ColName[:-1]: colVal}
        for word in rare_per_verb_dict.keys():
            if not word.isalpha():
                continue

            for col in rare_per_verb_dict[word]:
                try:
                    file = os.path.join(dir, word + "_" + col + ".txt")
                    with open(file, encoding='utf-8') as f:
                        lines = f.readlines()
                        for line in lines:
                            file.write("dep struct: ")
                            file.write(col)
                            new_f.write("\n")
                            new_f.write(line)

                except FileNotFoundError:
                    pass


if __name__ == '__main__':
    args = docopt.docopt(usage)
    #args["<output_path>"]
    if args["<csv_path>"]:
        Output = OutPutRowwise(args["<csv_path>"]
                           )
    else:
        Output = OutPutRowwise(args["<csv_path>"])
    if args["<perc>"]:
        Output.Output(args["<perc>"])
    else:
        Output.Output()
