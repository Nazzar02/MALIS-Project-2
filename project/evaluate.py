####################################################
# Code to evaluate the model thanks to the metrics #
####################################################

import json
import random
import pickle
import numpy as np
import pandas as pd
from numpy.linalg import norm
from tqdm import tqdm

#-------------------------------------------------------------------#

# paths
EMBEDDINGS_FILE = "embeddings/node2vec_embeddings_train.npy"
ALL_NODES_FILE = "embeddings/all_nodes_train.npy"
NODE2ID_FILE = "embeddings/node2id_train.pkl"
TEST_PLAYLISTS_JSON = "dataset/mpd.slice.9000-9999.json"
METRICS_OUTPUT_CSV = "evaluation_metrics.csv"

# Parameters
TEST_RATIO = 0.2  # hide 20% of song (Leave-N-Out)
K = 10            # Precision@K, Recall@K, MAP@K
TOP_NEW_CANDIDATES = 500
RANDOM_SEED = 42

#-------------------------------------------------------------------#


def cosine_similarity(v1, v2):
    denom = (norm(v1) * norm(v2))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(v1, v2) / denom)

def compute_embedding_mean(track_uris, embeddings, track2id):
    vectors = [embeddings[track2id[uri]] for uri in track_uris if uri in track2id]
    if not vectors:
        return np.zeros(embeddings.shape[1], dtype=np.float32)
    return np.mean(vectors, axis=0)


#-------------------------------------------------------------------#


def fairness(recommendations, user_hidden_tracks):
    fairness_scores = []
    for user_recs, user_hidden in zip(recommendations, user_hidden_tracks):
        user_fairness = len(set(user_recs) & set(user_hidden)) / len(user_hidden)
        fairness_scores.append(user_fairness)
    return np.mean(fairness_scores)

def coverage(recommendations, total_catalog):
    recommended_items = set(item for user_recs in recommendations for item in user_recs)
    return len(recommended_items) / len(total_catalog)

def novelty(recommendations, train_playlists):
    train_tracks = set(item for pl in train_playlists for item in pl)
    novel_items = sum(1 for user_recs in recommendations for item in user_recs if item not in train_tracks)
    total_recommendations = sum(len(user_recs) for user_recs in recommendations)
    return novel_items / total_recommendations

#-------------------------------------------------------------------#


def evaluate_group_recommendations():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED) #randomness

    # load
    embeddings = np.load(EMBEDDINGS_FILE)

    all_nodes = np.load(ALL_NODES_FILE, allow_pickle=True)

    with open(NODE2ID_FILE, "rb") as f:
        track2id = pickle.load(f)

    with open(TEST_PLAYLISTS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    test_playlists = data["playlists"]

    # Leave-N-Out we needed the help of chat gpt here to use this "new" method
    all_recommendations = []
    all_hidden_tracks = []
    all_visible_tracks = []
    catalog_tracks = set(track2id.keys())

    # eval for each user
    for pl in tqdm(test_playlists, desc="eval of playlist"):
        tracks_in_pl = [t["track_uri"] for t in pl["tracks"] if t.get("track_uri")]
        if len(tracks_in_pl) < 2:  # Skip if not enough tracks to split
            continue

        # separate between visible and inveisble tracks
        num_to_hide = max(1, int(TEST_RATIO * len(tracks_in_pl)))
        hidden_tracks = random.sample(tracks_in_pl, num_to_hide)
        visible_tracks = [t for t in tracks_in_pl if t not in hidden_tracks]
        all_hidden_tracks.append(hidden_tracks)
        all_visible_tracks.append(visible_tracks)

        # group user embeddings
        playlist_vector = compute_embedding_mean(visible_tracks, embeddings, track2id)

        # candidates 
        candidate_tracks = [uri for uri in track2id if uri not in visible_tracks]

        # cos sim
        similarities = []
        for uri in candidate_tracks:
            track_vector = embeddings[track2id[uri]]
            similarity = cosine_similarity(playlist_vector, track_vector)
            similarities.append((uri, similarity))

        # sort to keep only the best
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_recommendations = [uri for uri, _ in similarities[:TOP_NEW_CANDIDATES]]
        all_recommendations.append(top_recommendations)

    # metrics calculus
    fairness_score = fairness(all_recommendations, all_hidden_tracks)
    coverage_score = coverage(all_recommendations, catalog_tracks)
    novelty_score = novelty(all_recommendations, all_visible_tracks)

    # save on csv file
    metrics_df = pd.DataFrame({
        "Metric": ["Fairness", "Coverage", "Novelty"],
        "Score": [fairness_score, coverage_score, novelty_score]
    })
    metrics_df.to_csv(METRICS_OUTPUT_CSV, index=False)

    # Résultats
    print("\n======= Results =======")
    print(metrics_df)
    
#-------------------------------------------------------------------#

if __name__ == "__main__":
    evaluate_group_recommendations()
