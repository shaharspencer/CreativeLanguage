from sentence_transformers import SentenceTransformer, util




class SimilarityScore:
    def __init__(self, nlp):
        self.nlp = nlp
        self.sentence_model = SentenceTransformer(
            'sentence-transformers/all-mpnet-base-v2')

    def token_pair_similarity(self, first_token, second_token):
        """
              Compute similarity between two tokens.

              Args:
                  first_token (str): The first token.
                  second_token (str): The second token.

              Returns:
                  float: The similarity score.
        """
        spacy_token_1, spacy_token_2 = self.nlp(first_token), self.nlp(second_token)
        return spacy_token_1.similarity(spacy_token_2)

    def sentence_pair_similarity(self, sent_1, sent_2):
        """
        Compute similarity between two sentences.

        Args:
            sent_1 (str): The first sentence.
            sent_2 (str): The second sentence.

        Returns:
            float: The similarity score.
        """
        embedding_1 = self.sentence_model.encode(sent_1,
                                                 convert_to_tensor=True)
        embedding_2 = self.sentence_model.encode(sent_2,
                                                 convert_to_tensor=True)

        sim = util.pytorch_cos_sim(embedding_1, embedding_2)[0]
        return sim[0].item()

