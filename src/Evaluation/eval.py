from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import seaborn as sns
import matplotlib.pyplot as plt

class SimilarityNode:
    def __init__(self, comment_id, parent_comment_id, similarity_score):
        self.comment_id = comment_id
        self.parent_comment_id = parent_comment_id
        self.similarity_score = similarity_score
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def __repr__(self, level=0):
        ret = "\t" * level + f"Comment ID: {self.comment_id}, Level: {level}, Similarity: {self.similarity_score}\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret

class EvalSimilarity:

    def __init__(self, original_posttree, gen_posttree):

        self.org_posttree = original_posttree
        self.gen_posttree = gen_posttree
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

    def _build_tree(self, data):
       
        nodes = {}
        root_nodes = []

        for entry in data:
            node = SimilarityNode(entry["comment_id"], entry["parent_comment_id"], entry["similarity_score"])
            nodes[entry["comment_id"]] = node

        for entry in data:
            node = nodes[entry["comment_id"]]
            if not entry["parent_comment_id"]: 
                root_nodes.append(node)
            else:
                parent_node = nodes[entry["parent_comment_id"]]
                parent_node.add_child(node)

        return root_nodes
    
    def compare_comments(self):
        
        similarities = []

        for onode, gnode in zip(self.org_posttree.bfs_generator(), self.gen_posttree.bfs_generator()):
            og_similarity = {}
            
            og_similarity["comment_id"] = onode.comment_id
            og_similarity["parent_comment_id"] = onode.parent_comment_id
            og_similarity["similarity_score"] = self.check_cosinesim([onode.comment_text, gnode.comment_text])[0, 1]

            similarities.append(og_similarity)
        
        similaritree = self._build_tree(similarities)

        return similaritree