import pandas as pd
from docopt import docopt
import csv
from collections import OrderedDict

usage = '''
Processor CLI.
Usage:
    ExploreVerbDependencies.py
    ExploreVerbDependencies.py <csv_path>
    ExploreVerbDependencies.py <csv_path> <output_path>
'''

class DependencyFrequency:
    def __init__(self, source_csv = r"C:\Users\User\PycharmProjects\CreativeLanguage\explore_verb_patterns\Dependencies\deps_15000_posts.csv"):
        self.source_csv= source_csv


    def OutputToCsv(self, output_path = "dep_freqs_15000_sents.csv"):
        # open csv as dataframe
        df = pd.read_csv(self.source_csv)
        # get columns representing dep struct: name ends with COUNT
        count_dict = {}
        for colName, colCont in df.iteritems():
            if not colName.endswith("COUNT"):
                continue
            col_sum = colCont.sum()
            if colName == "V_COUNT":
                count_dict["NO_DEP"] = col_sum
            else:
                newColName = colName[:len(colName)-len("COUNT")-1]
                count_dict[newColName] = col_sum

        # sort columns by their sum
        totalSum = sum(count_dict.values())
        count_dict = sorted(count_dict.items(), key = lambda x: x[1], reverse=True)

        with open(output_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["dep_struct", "%of_total", "count"])
            for key, value in count_dict:
                writer.writerow([key, (value / totalSum), value])


if __name__ == '__main__':
    args = docopt(usage)
    if args["<csv_path>"]:
        dep = DependencyFrequency(args["<csv_path>"])
    else:
        dep = DependencyFrequency()
    if args["<output_path>"]:
        dep.OutputToCsv(args["<output_path>"])
    else:
        dep.OutputToCsv()
