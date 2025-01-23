#########################################
# Code to remove miscellaneous clusters #
#########################################

import os
import csv

def clean_clusters(input_file, output_file, threshold):
    with open(input_file, 'r', newline='', encoding='utf8') as infile, \
         open(output_file, 'w', newline='', encoding='utf8') as outfile:

        # The part with the csv readers/writers has been adapted from chatGPT
        reader = csv.DictReader(infile, delimiter=',')
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()

        #Filter the lines based on the threshold
        for row in reader:
            try:
                #convert the percentages in floats to avoid errors
                percentage = float(row["Exact Match Percentage"].replace('%', '').strip())
                if percentage > threshold:#filter
                    writer.writerow(row)
            except ValueError:
                continue

def main():
    input_file = "/home/vellard/malis/clustering-no-split/clusters/200/clusters_with_exact_matches.csv"
    output_file = "/home/vellard/malis/clustering-no-split/clean/200/clusters_clean.csv"

    clean_clusters(input_file, output_file, threshold=2)

    print(f"Clean clusters saved to {output_file}.")

if __name__ == "__main__":
    main()

