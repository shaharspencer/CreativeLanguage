import csv

import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
import faiss

from src.extract_creative_sentences_by_dim.Embeddings.bert_embeddings import \
    ContextualizedEmbeddings

# print(faiss.FAISS_VERSION_PATCH)

# print(faiss.__version__)

class FAISS:
    def __init__(self, df):
        dataset = Dataset.from_pandas(df)
        c = ContextualizedEmbeddings()
        embeddings_dataset = dataset.map(
            lambda x: {"embeddings": c.contextualized_embeddings(x["tokenized sentence"], x["token index"]).detach().numpy()}
        )
        vector_dimension = 768

        midpoint = np.mean(embeddings_dataset["embeddings"], axis=0)
        midpoint = midpoint.reshape(1, -1)

        index = faiss.IndexFlatL2(vector_dimension)

        assert index.is_trained

        index.add(np.asarray(embeddings_dataset["embeddings"]))

        n_total = index.ntotal


        D, I = index.search(midpoint, k=4227)
        file = open("eat_VERB_output_50_limit.csv", "w", encoding='utf-8', newline='')
        fields = ["sent_index", "sent", "distance"]
        writer = csv.DictWriter(f=file, fieldnames=fields)
        writer.writerow({"sent_index": "sent_index", "sent": "sent",
                         "distance": "distance"})
        for i, d in zip(I[0], D[0]):
            if len(embeddings_dataset[int(i)]["tokenized sentence"]) > 50:
                continue
            n_dict = {"sent_index": i,
                      "sent": embeddings_dataset[int(i)]["sentence"],
                      "distance": float(d)}
            writer.writerow(n_dict)
        file.close()



if __name__ == '__main__':
    dtypes = {
        'lemma': str,
        'word form': str,
        'sentence': str,
        'doc index': int,
        'sent index': int,
        'token index': int,
    }
    converters = {'tokenized sentence': eval,
                  'verb embeddings':eval}
    df = pd.read_csv('eat_VERB.csv', dtype=dtypes,converters=converters)
    datafiles = ["cool.csv"]
    f = FAISS(df)