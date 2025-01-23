# README

This project processes playlists to generate embeddings, tune hyperparameters, and evaluate recommendations. Below is the execution order and a brief explanation of each script:

1. **build_full_graph.py**: Constructs bipartite graphs (playlists and tracks).

2. **node2vec_embeddings.py**: Generates node embeddings using the Node2Vec algorithm.

3. **node2vec_embeddings_hyperparameter_tuning.py**: Tunes Node2Vec hyperparameters to optimize embeddings.

4. **recommender.py**: Generates track recommendations based on the group’s playlist preferences as group_recommendations.csv.

5. **evaluate.py**: Evaluates the quality of the recommendations using specifics metrics.
