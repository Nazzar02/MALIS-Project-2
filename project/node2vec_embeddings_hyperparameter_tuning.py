########################################################################################################
# Code to train the model node2vec with the bibarite graph generated and fond the best hyperparameters #
########################################################################################################

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
from sklearn.metrics.pairwise import cosine_similarity

#-------------------------------------------------------------------#

#paths
graph_path_train = "graphs/preferences_graph_train.pkl"  
graph_path_val = "graphs/preferences_graph_val.pkl"     
embeddings_path = "embeddings/node2vec_embeddings_train.npy" 
output_plot = "plots/embeddings_visualization.png"           

all_nodes_file = "embeddings/all_nodes_train.npy"
node2id_file = "embeddings/node2id_train.pkl"

# hyperparameters to test
walk_length_candidates = [5, 10, 15]
num_walks_candidates = [3, 5, 7]
vector_size_candidates = [32, 64, 128]

# hyperparameters
window = 5
workers = 4
epochs = 5  # very few epoch here, we just want fo find the bests hyperparameters

sample_size = 10000   # only for the visualisation with UMPA

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

#here we split train, evaluate and save as we are searching the best hyperparameters

def train_model(walks, all_nodes, vector_size=64, window=5, workers=4, epochs=5):
    model = Word2Vec(
        sentences=walks,
        vector_size=vector_size,
        window=window,
        min_count=1,
        sg=1,       # skip-gram
        workers=workers,
        epochs=epochs
    )
    # extrac embedding 
    embeddings = np.array([model.wv[str(n)] for n in all_nodes])
    return embeddings

#-------------------------------------------------------------------#

# Sample n_pairs real edges from G_val and n_pairs non-connected node pairs, 
# compare mean cosine similarity (real edges) vs (false pairs),                 #chat gpt helped find this strategie
# final score = (mean_sim_of_edges - mean_sim_of_non_edges).

def evaluate_embeddings(embeddings, all_nodes, G_val, n_pairs=10000):

    # mapping node -> index embedding
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}

    edges_val = list(G_val.edges())
    if len(edges_val) == 0: #empty case handdling
        return 0.0

    # sampling n_pairs edges 
    edges_val_sample = random.sample(edges_val, min(n_pairs, len(edges_val)))

    # Sample n_pairs pairs of nodes that are not connected randomly select 2 nodes and check that they do not form an edge
    non_edges_val_sample = []
    all_nodes_val = list(G_val.nodes())
    node_set_val = set(all_nodes_val)
    max_tries = 5 * n_pairs  # infinite case break
    tries = 0
    while len(non_edges_val_sample) < len(edges_val_sample) and tries < max_tries:
        a, b = random.sample(all_nodes_val, 2)
        if not G_val.has_edge(a, b):
            non_edges_val_sample.append((a, b))
        tries += 1

    # embeddings extracs for each pair of edges
    def pair_cos_sim(u, v):
        if (u not in node_to_idx) or (v not in node_to_idx):
            return 0.0  # if node from val is not in train
        emb_u = embeddings[node_to_idx[u]]
        emb_v = embeddings[node_to_idx[v]]
        sim = cosine_similarity([emb_u], [emb_v])
        return sim[0,0]

    #avg sim on edges
    sim_edges = [pair_cos_sim(u,v) for (u,v) in edges_val_sample]
    mean_sim_edges = np.mean(sim_edges) if sim_edges else 0.0

    sim_non_edges = [pair_cos_sim(u,v) for (u,v) in non_edges_val_sample]
    mean_sim_non_edges = np.mean(sim_non_edges) if sim_non_edges else 0.0

    score = mean_sim_edges - mean_sim_non_edges
    return score

#-------------------------------------------------------------------#

def hyperparam_search(G_train, G_val):
    #Tests different values of (walk_length, num_walks, vector_size) 
    # and returns the best combination based on the score from evaluate_embeddings().
    best_score = float("-inf")
    best_params = None
    best_embeddings = None
    all_nodes_train = list(G_train.nodes())

    #seach for each parameter walk_length, num_walks, vector_size
    for wl in walk_length_candidates:
        for nw in num_walks_candidates:
            for vs in vector_size_candidates:
                print(f"\n--- Test : walk_length={wl}, num_walks={nw}, vector_size={vs} ---")
                # Random walks
                train_walks, train_nodes = generate_random_walks(G_train, wl, nw)

                # training
                embeddings_tmp = train_model(
                    train_walks,
                    train_nodes,
                    vector_size=vs,
                    window=window,
                    workers=workers,
                    epochs=epochs 
                )

                # eval on G_val
                score = evaluate_embeddings(embeddings_tmp, train_nodes, G_val)
                print(f"Score on val = {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_params = (wl, nw, vs)
                    best_embeddings = embeddings_tmp

    print(f"\n=== best parameters ===")
    print(f"best score = {best_score:.4f}")
    print(f"best hyperparameters: walk_length={best_params[0]}, num_walks={best_params[1]}, vector_size={best_params[2]}")
    return best_params, best_embeddings

#-------------------------------------------------------------------#

def visualize_embeddings(embeddings, all_nodes, G, output_path, sample_size=10000):
    print("\nRéduction de dimension et visualisation...")
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
    print(f"  --> Réduction de dimension terminée en {end_time - start_time:.2f} secondes.\n")
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
    #load
    G_train = load_graph(graph_path_train)
    G_val = load_graph(graph_path_val)

    # hyperparameter search
    best_params, best_embeddings = hyperparam_search(G_train, G_val)
    wl_best, nw_best, vs_best = best_params

    # Re train with the best hyperparameters
    print("\n=== train with best hyperparameters ===")
    print(f"  -> walk_length={wl_best}, num_walks={nw_best}, vector_size={vs_best}")
    final_walks, final_nodes = generate_random_walks(G_train, wl_best, nw_best)
    final_model = Word2Vec(
        sentences=final_walks,
        vector_size=vs_best,
        window=window,
        min_count=1,
        sg=1,
        workers=workers,
        epochs=10  
    )
    final_embeddings = np.array([final_model.wv[str(n)] for n in final_nodes])

    #save
    np.save(embeddings_path, final_embeddings)
    np.save(all_nodes_file, np.array(final_nodes, dtype=object))
    node2id = {str(n): i for i, n in enumerate(final_nodes)}
    with open(node2id_file, "wb") as f:
        pickle.dump(node2id, f)

    # 5) Visualisation
    #visualize_embeddings(final_embeddings, final_nodes, G_train, output_plot, sample_size=sample_size)
    print("done")
