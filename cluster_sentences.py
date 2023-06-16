"""
    this class aims to measure distances
"""

from sentence_transformers import SentenceTransformer
import numpy as np
#https://huggingface.co/sentence-transformers/all-roberta-large-v1

class ClusterSentences:
    def __init__(self): #TODO find best model
        self.model = SentenceTransformer(
            'sentence-transformers/all-roberta-large-v1')
    """
        embedd all sentences 
    """
    #TODO i want embeddings to be connected to the sentence
    def embedd(self, sentences):
        embeddings = self.model.encode(sentences)
        return embeddings

    """
        this method finds the "centerpoint" of embeddings, represented
        by the mean embedding for these sentences
    """
    def find_equiblirum(self, embeddings):
        return np.mean(embeddings, axis=0)

    """
        find furthest vector from the equiblirum point
        we do this to find the normalization factor and normalize
        accords different words, by getting a scale of 0-1
    """
    def find_normalize_factor(self, embeddings, embed_average): #TODO what if there is only one
        furthest_distance, furthest_vector = float('-inf') #TODO define vector of all zeros
        for vector in embeddings:
            distance = np.linalg.norm(vector - embed_average)
            if distance > furthest_distance:
                furthest_distance = distance
                furthest_vector = vector

        return furthest_distance, furthest_vector


