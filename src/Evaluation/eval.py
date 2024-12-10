from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import seaborn as sns
import matplotlib.pyplot as plt

import os
from collections import deque, defaultdict


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

        print(sentences)

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

            # Ensure original comment_text is a string
            org_text = str(onode.comment_text) if onode.comment_text is not None else ""

            # Check generated comment_text: if it's a list, ensure all elements are strings; if single, cast to string
            if isinstance(gnode.comment_text, list):
                gen_texts = [str(t) if t is not None else "" for t in gnode.comment_text]
                sentences = [org_text] + gen_texts
                similarity_matrix = self.check_cosinesim(sentences)

                # The first sentence is the original comment; the rest are generated responses
                generated_similarities = similarity_matrix[0, 1:]  # Compare original (row=0) to others
                avg_similarity = np.mean(generated_similarities)
                og_similarity["similarity_score"] = avg_similarity
            else:
                # Single generated text
                gen_text = str(gnode.comment_text) if gnode.comment_text is not None else ""
                sentences = [org_text, gen_text]
                similarity_matrix = self.check_cosinesim(sentences)
                og_similarity["similarity_score"] = similarity_matrix[0, 1]

            similarities.append(og_similarity)

        similaritree = self._build_tree(similarities)
        return similaritree

    def bfs(self, root, levels, counts, bfs_file):
        """
        Perform BFS and calculate the cumulative similarity scores and counts at each level.
        """
        with open(bfs_file, "a") as file:
            queue = deque([(root, 0)])

            while queue:
                current, depth = queue.popleft()

                levels[depth] += current.similarity_score
                counts[depth] += 1

                indent = "    " * depth
                line = (f"{indent}Parent ID: {current.parent_comment_id}, "
                        f"Comment ID: {current.comment_id}, "
                        f"Similarity: {current.similarity_score:.2f}")
                print(line)
                file.write(line + "\n")

                for child in current.children:
                    queue.append((child, depth + 1))