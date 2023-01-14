import pandas as pd
from docopt import docopt
import csv
from collections import OrderedDict
import utils.path_configurations as paths
import os

usage = '''
Processor CLI.
Usage:
    sort_structures_by_frequency.py <source_csv> <num_posts>
'''

class DependencyFrequency:
    def __init__(self, source_csv, num_posts):
        self.num_posts = num_posts
        self.source_csv = os.path.join(paths.files_directory,
                                   paths.dependency_set_directory,
                                   paths.dependency_set_source_files,
                                   source_csv)



    def OutputToCsv(self, output_path):
        # open csv as dataframe
        output_path = os.path.join(paths.files_directory,
                                   paths.dependency_set_directory,
                                   paths.dependency_set_source_files,
                                   output_path)

        df = pd.read_csv(self.source_csv)
        # get columns representing dep struct: name ends with COUNT
        count_dict = {}
        for colName, colCont in df.iteritems():
            if not colName.endswith("COUNT"):
                continue
            col_sum = colCont.sum()

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
    from datetime import datetime
    args = docopt(usage)
    dep = DependencyFrequency(source_csv=args["<source_csv>"],
                              num_posts=args["<num_posts>"])
    datetime = datetime.today().strftime('%Y_%m_%d')
    dep.OutputToCsv("dep_freqs_first_{n}_posts_".format(n=args["<num_posts>"]
                                                       ) + datetime + ".csv")


