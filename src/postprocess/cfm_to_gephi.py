import csv

import numpy as np

def to_gephi(cfm, output_fp):
    """
    Converting Confusion Matrix in 2-D Matrix Format (List of Lists) to Gephi Readable .csv
    """
    dimensions = len(cfm)
    with open(output_fp,'w') as readable_csv:
        first_row = ""
        for i in range(dimensions):
            first_row = first_row + "," + str(i)
        readable_csv.write(first_row+"\n")
        for i in range(dimensions):
            row = str(i)
            for item in cfm[i]:
                row = row + "," + str(item)
            readable_csv.write(row+"\n")