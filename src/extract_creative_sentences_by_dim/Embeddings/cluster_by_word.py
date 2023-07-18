import pandas as pd
from datasets import Dataset

from src.extract_creative_sentences_by_dim.Embeddings.bert_embeddings import \
    ContextualizedEmbeddings


class ClusterSentences:
    def __init__(self): #TODO find best model
        self.embedder = ContextualizedEmbeddings()

    def process_csv(self, csv_name): #NOT ENOUGH MEMORY
        dtypes = {
            'lemma': str,
            'word form': str,
            'sentence': str,
            'doc index': int,
            'sent index': int,
            'token index': int,
        }
        converters = {'tokenized sentence': eval,
                      'verb embeddings': eval}
        csv = pd.read_csv(csv_name, encoding='utf-8', dtype=dtypes,
                          converters=converters)
        dataset = Dataset.from_pandas(csv)
        embeddings_dataset = dataset.map(
            lambda x: {"embeddings": self.embedder.contextualized_embeddings(
                x["tokenized sentence"], x["token index"]).detach().numpy()}
        )
        equilblirum = self.__find_equiblirum(embeddings_dataset["embeddings"])
        print(f"equiblirum is {equilblirum}\n")
        normal_factor, furthest_vector = self.__find_normalize_factor(embeddings_dataset["embeddings"], equilblirum)
        print(f"")
        normalized_embeddings = embeddings_dataset.map()
        normalized_embeddings = self.__normalize_embeddings(embeddings, normal_factor)

        csv['embedding distance'] = list(map(lambda x: np.linalg.norm(x-
                                                            equilblirum),
                                             normalized_embeddings)) #TODO VECTORIZE
        csv.to_csv('output.csv', index=False)


    """
        embedd all sentences 
    """
    #TODO i want embeddings to be connected to the sentence
    def __embedd(self, sentences):
        embeddings = self.model.encode(sentences, batch_size=5000)
        return embeddings

    """
        this method finds the "centerpoint" of embeddings, represented
        by the mean embedding for these sentences
    """
    def __find_equiblirum(self, embeddings):
        return np.mean(embeddings, axis=0)

    """
        find furthest vector from the equiblirum point
        we do this to find the normalization factor and normalize
        accords different words, by getting a scale of 0-1
        @:param embeddings ndarray[tensor] of n tensors the size of a sentence emedding
        @:param embed_average 1*d tensor the average embedding vector
    """
    def __find_normalize_factor(self, embeddings, embed_average): #TODO what if there is only one
        furthest_distance, furthest_vector = float('-inf'), None #TODO define vector of all zeros
        for vector in embeddings:
            distance = np.linalg.norm(vector - embed_average)
            if distance > furthest_distance:
                furthest_distance = distance
                furthest_vector = vector

        return furthest_distance, furthest_vector

    def __normalize_embeddings(example)->list[np.ndarray]:
        # convert the list of tensors to a single tensor
        # tensor_array = torch.stack(embeddings)

        # divide all tensors by the normalization factor
        processed_array = embeddings / normalize_factor

        # convert the processed array back to a list of tensors
        processed_list = list(processed_array)

        return processed_list


if __name__ == '__main__':

    cluster = ClusterSentences()
    cluster.process_csv("find_VERB.csv")