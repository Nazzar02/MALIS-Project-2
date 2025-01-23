######################################################################
# Code to generate group_recommendations.csv using the model trained #
######################################################################

import os
import json
import math
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

#-------------------------------------------------------------------#

# Paths
EMBEDDINGS_FILE = "embeddings/node2vec_embeddings_train.npy"
ALL_NODES_FILE = "embeddings/all_nodes_train.npy"
NODE2ID_FILE = "embeddings/node2id_train.pkl"

TRACKS_CSV = "tracks.csv"

GROUP_PLAYLISTS_JSON = "input.json"

OUTPUT_CSV = "group_recommendations.csv"

# Parameters
MIN_NEW_RATIO = 0.2 #20% of new songs
MIN_TOTAL_LINES = 30 # min 30 songs
TOP_NEW_CANDIDATES = 300 # song choosen among 300 candidates

#-------------------------------------------------------------------#

def check_files_exist(*files):
    for file in files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"no file : {file}")  #error case

#-------------------------------------------------------------------#

# calculate sim (v1,v2)
def cosine_similarity(v1, v2):
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(v1, v2) / denom)

# calculate the mean embedding of the group
def compute_group_embedding(track_uris, embeddings, track2id):
    valid_vectors = []
    for uri in track_uris:
        if uri in track2id:
            idx = track2id[uri]
            valid_vectors.append(embeddings[idx])
    if not valid_vectors


#-------------------------------------------------------------------#


def generate_group_recommendations(
    group_json, tracks_csv, embeddings_file, all_nodes_file,
    node2id_file, output_csv, min_new_ratio, min_total_lines, top_new_candidates
):

    # file check
    check_files_exist(group_json, tracks_csv, embeddings_file, all_nodes_file, node2id_file)

    # load
    embeddings = np.load(embeddings_file)
    all_nodes = np.load(all_nodes_file, allow_pickle=True)
    with open(node2id_file, "rb") as f:
        track2id = pickle.load(f)

    tracks_df = pd.read_csv(tracks_csv)     #using a pdf to get the recommanded playlist 
    track_info = {
        row.track_uri: {
            "track_name": row.track_name,
            "artist_name": row.artist_name,
            "album_name": getattr(row, "album_name", None)
        }
        for row in tracks_df.itertuples(index=False)
    }

    with open(group_json, "r", encoding="utf-8") as f:
        group_playlists = json.load(f)["playlists"]

    # Extraction songs of the group
    track_occurrences = defaultdict(int)
    for pl in group_playlists:
        unique_tracks = set(t["track_uri"] for t in pl["tracks"])
        for uri in unique_tracks:
            track_occurrences[uri] += 1

    group_tracks = list(track_occurrences.keys())
    group_vector = compute_group_embedding(group_tracks, embeddings, track2id)

    # songs already in the groups's preference
    in_group_rows = [
        {
            "Track Name": track_info.get(uri, {}).get("track_name", uri),
            "Artist": track_info.get(uri, {}).get("artist_name", "Unknown Artist"),
            "Occurrences": occ,
            "Source Playlist": f"Input (shared by {occ})"
        }
        for uri, occ in track_occurrences.items()
    ]

    # similarity for NEW songs
    candidate_tracks = [t for t in track2id if t not in track_occurrences]

    new_sims = []
    for t_uri in tqdm(candidate_tracks, desc="Calcul of similarities"):
        idx = track2id[t_uri]
        sim = cosine_similarity(group_vector, embeddings[idx])
        new_sims.append((t_uri, sim))

    new_sims.sort(key=lambda x: x[1], reverse=True)
    top_new_sims = new_sims[:top_new_candidates]

    new_rows_all = [        #same logic as above
        {
            "Track Name": track_info.get(uri, {}).get("track_name", uri),
            "Artist": track_info.get(uri, {}).get("artist_name", "Unknown Artist"),
            "Occurrences": 0,
            "Source Playlist": "Other Input"
        }
        for uri, _ in top_new_sims
    ]

    # final file construction
    needed_new = max(math.ceil(min_new_ratio * len(in_group_rows)), min_total_lines - len(in_group_rows))
    needed_new = min(needed_new, len(new_rows_all))

    final_rows = in_group_rows + new_rows_all[:needed_new]
    while len(final_rows) < min_total_lines and needed_new < len(new_rows_all):
        final_rows.append(new_rows_all[needed_new])
        needed_new += 1

    # save
    df = pd.DataFrame(final_rows, columns=["Track Name", "Artist", "Occurrences", "Source Playlist"])
    df.to_csv(output_csv, index=False, sep=";")

#-------------------------------------------------------------------#

if __name__ == "__main__":
    generate_group_recommendations(
        GROUP_PLAYLISTS_JSON, TRACKS_CSV, EMBEDDINGS_FILE,
        ALL_NODES_FILE, NODE2ID_FILE, OUTPUT_CSV,
        MIN_NEW_RATIO, MIN_TOTAL_LINES, TOP_NEW_CANDIDATES
    )
    print ("done")
