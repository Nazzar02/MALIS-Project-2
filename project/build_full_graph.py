##############################################################################
# Code to generate bipartite graph from Json database to be used by node2vec #
##############################################################################

import json
import networkx as nx
import pickle
import os

#-------------------------------------------------------------------#

# We use a 80-10-10 split fllowing the natural contruction of the slides
slices_config = {
    "train": range(0, 8000, 1000),  # Slices 0-7999
    "val": range(8000, 9000, 1000),  # Slices 8000-8999
    "test": range(9000, 10000, 1000)  # Slices 9000-9999
}


#I/O dir
input_dir = "dataset"
output_dir = "graphs"
os.makedirs(output_dir, exist_ok=True)


#-------------------------------------------------------------------#

#function that will be used in node2vec_embeddings.py

def build_graph(slice_ids, output_path):

    G = nx.Graph()
    total_playlists = 0
    total_tracks = 0

    for start_id in slice_ids:
        slice_path = os.path.join(input_dir, f"mpd.slice.{start_id}-{start_id + 999}.json") #get the slices

        if not os.path.exists(slice_path):
            print(f"---No slice: {slice_path}---")
            continue

        # upload the Json files
        with open(slice_path, "r") as f:
            data = json.load(f)

        for playlist in data["playlists"]:          #add tracks to the graph
            pid = f"playlist_{playlist['pid']}"
            G.add_node(pid, type="playlist")
            total_playlists += 1

            for track in playlist["tracks"]:          #add playlist to the graph
                tid = track["track_uri"]
                if not G.has_node(tid):
                    G.add_node(tid, type="track")
                    total_tracks += 1
                G.add_edge(pid, tid, weight=1)

    # save in pickle file
    with open(output_path, "wb") as f:
        pickle.dump(G, f)


# make the graoh for train, val and test
for phase, slice_ids in slices_config.items():
    output_path = os.path.join(output_dir, f"preferences_graph_{phase}.pkl")
    build_graph(slice_ids, output_path)
