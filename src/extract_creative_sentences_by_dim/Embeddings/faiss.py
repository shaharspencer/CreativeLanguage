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

        x = 0
        index = faiss.IndexFlatL2(vector_dimension)
        index.add(n=embeddings_dataset["embeddings"], x=1)
        if not index.is_trained:
            pass
        # index.add(embeddings_dataset["embeddings"])



        # res = index.search(midpoint)
        x = 0




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
    df = pd.read_csv('cool.csv', dtype=dtypes,converters=converters)
    datafiles = ["cool.csv"]
    f = FAISS(df)