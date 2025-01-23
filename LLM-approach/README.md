### README

## 1. Transform and pre-process the dataset
Run:
```bash
python3 transform-dataset/json2csv.py
```
to transform the dataset to json files to user-readable csv files.

## 2. Embedding generation and clustering
First, playlists titles and tracks (of 10 slices) are embedded using a pre-trained SentenceBERT model and stored in /embeddings/embeddings.pkl:
```bash
python3 clustering-no-split/embeddings/track_embeddings_no-split.py
```

Then, the K-means clustering algorithm is applied to create the clusters, and the generated csv file (in /clustering-no-split/clusters/200) is modified to calculate and include the percentage of exact matches:
```bash
python3 clustering-no-split/clusters/clustering-no-split.py clustering-no-split/clusters/percent-no-split.py
```
The resulted csv file is stored into clustering-no-split/clusters/200/clusters_with_exact_matches.csv

Apply the clean algorithm to remove miscellaenous clusters and save the output in clustering-no-split/clean/200:
```bash
python3 clustering-no-split/clean/clean.py
```
Finally, randomly split the clusters:
```bash
python3 clustering-no-split/split/split.py
```
## 3. Finetuning (abandoned)
Run:
```bash
python3 finetuning/finetuning-new.py
```
to train the sentenceBERT model and store the configuration in fientuning/sentence-bert/.
## 4. Generate the embeddings for playlists titles (preferably using the fine-tuned model, otherwise a pre-trained model)
run :
```bash
python3 embeddings/sbert_embeddings.py
```
saves a picke file in embeddings/sbert_playlist_embeddings.pkl

## 5. Generate the grouped playlist
Run:
```bash
python3 similarity/test-model.py
```

