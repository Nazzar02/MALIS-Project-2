##############################################################################
# Code to compute the embeddings of playlists using pre-trained sentenceBERT #
##############################################################################


import csv
import pickle
from sentence_transformers import SentenceTransformer


def playlist_embeddings(input_csv, output_pickle, model_name="paraphrase-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)

    #Dictionary to store the embeddings
    playlist_embeddings = {}

    with open(input_csv, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            playlist_id = row["pid"]
            playlist_name = row["name"]
            
            embedding = model.encode(playlist_name, convert_to_numpy=True)#encore() to creqte the embeddings

            playlist_embeddings[playlist_id] = {
                "name": playlist_name,
                "embedding": embedding
            }#the dictionary contains the playlist title as the key and its embedding as the value

#----------------------------------------------#
#Save in a pickle file

    with open(output_pickle, 'wb') as f:
        pickle.dump(playlist_embeddings, f)


playlist_embeddings(
    input_csv="/data/playlist_continuation_data/csvs/playlists.csv",
    output_pickle="/home/vellard/malis/embeddings/sbert_playlists_embeddings.pkl",
    model_name="paraphrase-MiniLM-L6-v2"
)
