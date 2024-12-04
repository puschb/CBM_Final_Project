from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import seaborn as sns
import matplotlib.pyplot as plt

class EvalSimilarity:

    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    def check_cosinesim(self, sentences):
        """
        input:
        sentences: list(str), can be pairwise.
        returns a dict with (sentence1, sentence2):cosine_similarity
        """

        if len(sentences) == 1:
            return np.array([[1.0]])

        embeddings = self.model.encode(sentences)

        # Compute and return cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix
    
    def plot_simheatmap(self, similarity_matrix, sentences):
        plt.figure(figsize=(7, 5))
        sns.heatmap(
            similarity_matrix, 
            xticklabels=sentences, 
            yticklabels=sentences, 
            cmap="coolwarm", 
            annot=True, 
            fmt=".2f", 
            cbar=True
        )
        plt.title("Cosine Similarity Heatmap")
        plt.xlabel("Sentences")
        plt.ylabel("Sentences")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()