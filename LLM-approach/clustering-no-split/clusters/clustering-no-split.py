###################
# Code to cluster #
###################

import os
import csv
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

#----------------------------------------#
#Apply K-means and write the clusters in a csv file

def cluster_playlists(playlist_embeddings, num_clusters, playlist_titles, playlist_tracks, output_file):
    embedding_matrix = np.array(list(playlist_embeddings.values()))
    pids = list(playlist_embeddings.keys())

    #K-means algorithm (pre-existing fucnctions)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto')
    cluster_labels = kmeans.fit_predict(embedding_matrix) #No PCA

    #csv file
    with open(output_file, 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(["Cluster ID", "Playlist ID", "Playlist Title", "Tracks"])#Header
        for pid, label in tqdm(zip(pids, cluster_labels), total=len(pids), desc="Clustering", unit="playlist"):
            writer.writerow([label, pid, playlist_titles.get(pid, ""), ";".join(playlist_tracks.get(pid, []))])

def main():
    embeddings_file = "/home/vellard/malis/embeddings/embeddings.pkl"
    output_file = "/home/vellard/malis/clustering-no-split/clusters/200/clusters.csv"
    os.makedirs(output_dir, exist_ok=True)

    with open(embeddings_file, 'rb') as f:
        data = pickle.load(f)#Load the pre-computed embeddings for a picle file

    playlist_embeddings = data["playlist_embeddings"]
    playlist_titles = data["playlist_titles"]
    playlist_tracks = data["playlist_tracks"]

    cluster_playlists(playlist_embeddings, num_clusters=200, playlist_titles=playlist_titles, playlist_tracks=playlist_tracks, output_file=output_file)
    
if __name__ == "__main__":
    main()
