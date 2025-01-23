#########################################
# Code to generate the grouped playlist #
#########################################

import os
import torch
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import csv
import json
import random

#------------------------------------------#
#Load the pre-trained sBERT model

def load_model(model_name="paraphrase-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    model.eval()
    return model

#------------------------------------------#
#Get the precomputed embeddings

def get_playlist_embedding(playlist_name, model):
    embedding = model.encode(playlist_name, convert_to_numpy=True)
    return embedding

#Split playlists into visible and hidden data
# Leave-n-out strategy proposed by chatGPT
def split_visible_and_hidden_data(input_playlists_data, split_ratio=0.8):
    visible_tracks_dict = {}
    hidden_tracks_dict = {}

    for playlist in input_playlists_data["playlists"]:
        tracks = playlist["tracks"]
        random.shuffle(tracks)  # Shuffle the tracks for randomness
        split_point = int(len(tracks) * split_ratio)

        # Split tracks
        visible_tracks = tracks[:split_point]
        hidden_tracks = tracks[split_point:]

        # Store them
        visible_tracks_dict[playlist["pid"]] = [track["track_name"].lower() for track in visible_tracks]
        hidden_tracks_dict[playlist["pid"]] = [track["track_name"].lower() for track in hidden_tracks]

    return visible_tracks_dict, hidden_tracks_dict

#-----------------------------------------------------#
#Select relevant songs

# Find the most similar playlists
def find_similar_playlists(input_playlists, playlist_embeddings, model, top_k=100):
    input_embeddings = [
        get_playlist_embedding(playlist, model) for playlist in input_playlists
    ]
    similarities = []
    for pid, metadata in playlist_embeddings.items():
        similarity = max(
            cosine_similarity([input_emb], [metadata["embedding"]])[0][0]
            for input_emb in input_embeddings
        )
        similarities.append((pid, similarity))

    sorted_playlists = sorted(similarities, key=lambda x: x[1], reverse=True)
    print(f"Similar playlists: {sorted_playlists[:top_k]}")
    return sorted_playlists[:top_k]

#Recommend the most occurent songs
def get_top_songs(similar_playlists, playlist_tracks, input_tracks_dict, final_size=30, input_min_ratio=0.2, input_weight=2):
    track_counter = Counter()
    input_track_counter = Counter()

    # Count occurrences of tracks
    for pid, _ in similar_playlists:
        if pid in playlist_tracks:
            for track in playlist_tracks[pid]:
                track_name = track["track_name"].lower()
                track_id = (track["track_name"], track["artist_name"])
                track_counter[track_id] += 1  #Count for all tracks

                #Count input tracks separately with weights according to the occurence
                if track_name in input_tracks_dict:
                    input_track_counter[track_id] += input_weight

    #To ensure a min ratio of input songs
    input_min_count = int(final_size * input_min_ratio)

    #Select the most occurent songs
    input_tracks = input_track_counter.most_common(input_min_count)
    input_tracks_set = set([track for track, _ in input_tracks])

    #fill the rest of the playlist with songs from the MDP
    remaining_tracks = [
        (track, count) for track, count in track_counter.most_common()
        if track not in input_tracks_set
    ]
    new_tracks = remaining_tracks[:final_size - len(input_tracks)]

    combined_tracks = input_tracks + new_tracks

    #Find the origin of each track (from input playlists or not)
    track_origin = {}
    for track, count in input_tracks:
        track_origin[track] = "Input"
    for track, count in new_tracks:
        track_origin[track] = "Other Input"

    return [(track, count, track_origin[track]) for track, count in combined_tracks]

#Generate the csv file playlist (adapted from chatGPT)
def generate_and_save_playlist(input_playlists, input_tracks_dict, playlist_embeddings, playlist_tracks, model, output_csv, final_size=100, top_similar=100):
    similar_playlists = find_similar_playlists(input_playlists, playlist_embeddings, model, top_similar)
    top_songs = get_top_songs(similar_playlists, playlist_tracks, input_tracks_dict, final_size)

    output_dir = os.path.dirname(output_csv)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_csv, 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(["Track Name", "Artist", "Occurrences", "Source Playlist"])
        for (track_name, artist), count, origin in top_songs:
            writer.writerow([track_name, artist, count, origin])

#################
# Main Function #
#################
def main():
    model_name = "paraphrase-MiniLM-L6-v2"  # Pre-trained model
    playlist_embeddings_file = "/home/vellard/malis/similarity/new_playlist_embeddings.pkl"
    items_csv = "/data/playlist_continuation_data/csvs-3/items.csv"
    tracks_csv = "/data/playlist_continuation_data/csvs-3/tracks.csv"
    input_file = "/home/vellard/malis/similarity/input/input.json"
    output_csv = "/home/vellard/malis/similarity/generated_playlist.csv"

    model = load_model(model_name)

    with open(playlist_embeddings_file, 'rb') as f:
        playlist_embeddings = pickle.load(f)

    with open(input_file, 'r', encoding='utf8') as f:
        input_playlists_data = json.load(f)

    visible_tracks_dict, hidden_tracks_dict = split_visible_and_hidden_data(input_playlists_data, split_ratio=0.8)

    visible_tracks_flattened = {
        track_name: pid
        for pid, tracks in visible_tracks_dict.items()
        for track_name in tracks
    }

    input_playlists = [playlist["name"] for playlist in input_playlists_data["playlists"]]
    generate_and_save_playlist(input_playlists, visible_tracks_flattened, playlist_embeddings, {}, model, output_csv, final_size=100)

    with open(output_csv, 'r', encoding='utf8') as f:
        reader = csv.reader(f)
        next(reader)  #(skip the reader)
        generated_playlist = [(row[0].lower(), row[1]) for row in reader]

#------------------------------------------#
#calculate the metrics

    hidden_tracks_set = {track for tracks in hidden_tracks_dict.values() for track in tracks}
    recommended_tracks_set = {track[0] for track in generated_playlist}

    #Chosen metrics
    fairness = len(hidden_tracks_set & recommended_tracks_set) / len(hidden_tracks_set) if hidden_tracks_set else 0
    print(f"Fairness: {fairness:.4f}")

    coverage = len(recommended_tracks_set & visible_tracks_flattened.keys()) / len(visible_tracks_flattened) if visible_tracks_flattened else 0
    print(f"Coverage: {coverage:.4f}")

    novelty = len(recommended_tracks_set - visible_tracks_flattened.keys()) / len(recommended_tracks_set) if recommended_tracks_set else 0
    print(f"Novelty : {novelty:.4f}")


if __name__ == "__main__":
    main()
