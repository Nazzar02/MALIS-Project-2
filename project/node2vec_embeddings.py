######################################################################
# Code to train the model node2vec with the bibarite graph generated #
######################################################################

import time
import pickle
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from umap import UMAP  # (better result than t-SNE)

#-------------------------------------------------------------------#


graph_path = "graphs/preferences_graph_train.pkl"  
embeddings_path = "embeddings/node2vec_embeddings_train.npy"  
loss_plot_path = "plots/loss_plot.png"  # was initially use but failed to make it work with Gensim
output_plot = "plots/embeddings_visualization.png"  

all_nodes_file = "embeddings/all_nodes_train.npy"
node2id_file = "embeddings/node2id_train.pkl"

# hyperparameters
walk_length = 15
num_walks = 7
vector_size = 64
window = 5
epochs = 10
workers = 4

sample_size = 10000  # only for the visualisation with UMPA

#-------------------------------------------------------------------#

def load_graph(graph_path):
    with open(graph_path, "rb") as f:
        G = pickle.load(f)  #get from pickle file
    return G

#-------------------------------------------------------------------#

def generate_random_walks(G, walk_length, num_walks):
    all_nodes = list(G.nodes())
    walks = []          #exemple of walks: [['1', '2', '5', '3'],['2', '3', '1', '4'],]
    for node in tqdm(all_nodes, desc="Random walks"):
        for _ in range(num_walks):                      #GPT's tips, use "_" for a "for" loop when we don't need the incrementer 
            walk = [str(node)]
            for _ in range(walk_length - 1):
                neighbors = list(G.neighbors(walk[-1]))
                if not neighbors:
                    break
                walk.append(str(random.choice(neighbors)))
            walks.append(walk)
    return walks, all_nodes

#-------------------------------------------------------------------#

def train_and_save_embeddings(walks, all_nodes, embeddings_path, all_nodes_file, node2id_file):
    start_time = time.time()
    model = Word2Vec(
        walks,
        vector_size=vector_size,
        window=window,
        min_count=1,
        sg=1,
        workers=workers,
        epochs=epochs
    )
    end_time = time.time()
    print(f" {end_time - start_time:.2f} s.\n")

    #saving the embeddings
    embeddings = np.array([model.wv[str(n)] for n in all_nodes])
    np.save(embeddings_path, embeddings)

    #saving all_nodes and node2id
    np.save(all_nodes_file, np.array(all_nodes, dtype=object))
    node2id = {str(n): i for i, n in enumerate(all_nodes)}  # maps each node to a unique int
    with open(node2id_file, "wb") as f:
        pickle.dump(node2id, f)
    return embeddings

#-------------------------------------------------------------------#

def visualize_embeddings(embeddings, all_nodes, G, output_path, sample_size=10000):
    start_time = time.time()

    # sub sampling because otherwise it is too long
    if len(embeddings) > sample_size:
        sampled_indices = np.random.choice(range(len(embeddings)), size=sample_size, replace=False)
        sampled_embeddings = embeddings[sampled_indices]
        sampled_nodes = [all_nodes[i] for i in sampled_indices]
    else:
        sampled_embeddings = embeddings
        sampled_nodes = all_nodes

    # PCA
    pca = PCA(n_components=2)
    reduced_pca = pca.fit_transform(sampled_embeddings)

    # UMAP
    umap = UMAP(n_components=2, random_state=42)
    reduced_umap = umap.fit_transform(sampled_embeddings)

    end_time = time.time()
    print(f"  --> plot in {end_time - start_time:.2f} s\n")

    plt.figure(figsize=(12, 6))

    #for visualization bellow I used chat GPT to generate this:

    # PCA
    plt.subplot(1, 2, 1)
    for node_type, color in zip(["playlist", "track"], ["red", "blue"]):
        indices = [i for i, n in enumerate(sampled_nodes) if G.nodes[n].get("type") == node_type]
        if indices:
            plt.scatter(
                reduced_pca[indices, 0], reduced_pca[indices, 1],
                c=color, label=node_type, s=10, alpha=0.6
            )
    plt.title("PCA Visualization")
    plt.xlabel("PCA Dim 1")
    plt.ylabel("PCA Dim 2")
    plt.legend()

    # UMAP
    plt.subplot(1, 2, 2)
    for node_type, color in zip(["playlist", "track"], ["red", "blue"]):
        indices = [i for i, n in enumerate(sampled_nodes) if G.nodes[n].get("type") == node_type]
        if indices:
            plt.scatter(
                reduced_umap[indices, 0], reduced_umap[indices, 1],
                c=color, label=node_type, s=10, alpha=0.6
            )
    plt.title("UMAP Visualization")
    plt.xlabel("UMAP Dim 1")
    plt.ylabel("UMAP Dim 2")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

#-------------------------------------------------------------------#

if __name__ == "__main__":

    G_train = load_graph(graph_path)

    # Random walks and train
    train_walks, train_nodes = generate_random_walks(G_train, walk_length, num_walks)
    train_embeddings = train_and_save_embeddings(train_walks, train_nodes, embeddings_path, all_nodes_file, node2id_file)

    # Viz
    visualize_embeddings(train_embeddings, train_nodes, G_train, output_plot, sample_size=sample_size)  #if too long just comment this line
    print("done")
