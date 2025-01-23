import os
import csv
import random
from os import path
from tqdm import tqdm

#for reproducibility of results
random.seed(1)

input_clusters_file = "/home/vellard/malis/clustering-no-split/clean/200/clusters_with_exact_matches.csv"
output_clusters_train = "/home/vellard/malis/clustering-no-split/split/clusters_train.csv"
output_clusters_val = "/home/vellard/malis/clustering-no-split/split/clusters_val.csv"
output_clusters_test = "/home/vellard/malis/clustering-no-split/split/clusters_test.csv"

# Load cluster data
clusters = {}

with open(input_clusters_file, 'r', newline='', encoding='utf8') as clusters_file:
    clusters_reader = csv.DictReader(clusters_file)
    headers = clusters_reader.fieldnames

    for row in clusters_reader:
        cluster_id = row["Cluster ID"]
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(row)
print(f"Total number of clusters: {len(clusters)}")

#output files
train_file = open(output_clusters_train, 'w', newline='', encoding='utf8')
val_file   = open(output_clusters_val,   'w', newline='', encoding='utf8')
test_file  = open(output_clusters_test,  'w', newline='', encoding='utf8')
train_writer = csv.DictWriter(train_file, fieldnames=headers)
val_writer   = csv.DictWriter(val_file,   fieldnames=headers)
test_writer  = csv.DictWriter(test_file,  fieldnames=headers)
train_writer.writeheader()
val_writer.writeheader()
test_writer.writeheader()

#80/10/10 split
for cluster_id, rows in clusters.items():
    random.shuffle(rows)
    nb_total = len(rows)

    #Calculate how many go to val and test
    nb_val  = int(0.1 * nb_total)
    nb_test = int(0.1 * nb_total)
    nb_train = nb_total - nb_val - nb_test

    #split
    train_rows = rows[:nb_train]
    val_rows   = rows[nb_train : nb_train + nb_val]
    test_rows  = rows[nb_train + nb_val : ]

    #Write in the output files
    for r in train_rows:
        train_writer.writerow(r)
    for r in val_rows:
        val_writer.writerow(r)
    for r in test_rows:
        test_writer.writerow(r)

# Close files
train_file.close()
val_file.close()
test_file.close()

print(f"Split clusters saved.")

