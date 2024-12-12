import os
from collections import deque, defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import f_oneway, kruskal, ttest_rel, wilcoxon
from scipy.stats import sem

from sentence_transformers import SentenceTransformer

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import networkx as nx

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

class PlotSimilarity:

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.all_files = self._get_all_files()

    def _get_all_files(self):
        all_files = os.listdir(self.folder_path)
        files = [
            os.path.join(self.folder_path, f)
            for f in all_files
            if f.endswith('.txt') and not f.endswith('_results.txt')
        ]
        return files
    
    def build_similarity_tree(self, file_path):
        """
        Given a file path, build a similarity tree of type SimilarityNode.
        """
        
        nodes = {}  
        roots = []

        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(", ")
                parent_id = parts[0].split(": ")[1].strip() if "Parent ID" in parts[0] else None
                comment_id = parts[1].split(": ")[1].strip()
                similarity = float(parts[2].split(": ")[1])

                node = SimilarityNode(comment_id, parent_id, similarity)
                nodes[comment_id] = node

                if parent_id and parent_id != "None": 
                    if parent_id in nodes:
                        nodes[parent_id].add_child(node)
                else:
                    roots.append(node)

        return roots[0] if len(roots) == 1 else roots
    
    def get_deepest_tree(self):
        
        """
        returns a list of roots of the deepest trees
        we do N runs per post, so len(roots) = N.
        """

        def calculate_depth(node):
            if not node.children:
                return 1
            return 1 + max(calculate_depth(child) for child in node.children)

        deepest_tree_roots = []

        files = self.all_files

        # iterate over each file in the folder.
        for file in files:
            
            # buiild the sim tree first,
            roots = self.build_similarity_tree(file)

            if isinstance(roots, SimilarityNode):
                roots = [roots]

            deepest_tree = None
            max_depth = 0

            for root in roots:
                depth = calculate_depth(root)
                if depth > max_depth:
                    max_depth = depth
                    deepest_tree = root
            
            deepest_tree_roots.append(deepest_tree)

        return deepest_tree_roots
    
    def calculate_avg_similarity(self, trees):
        """
        Takes in a list of SimilarityNode trees, and 
        returns another SimilarityNode tree with the avg scores,
        along with a dataframe containing all sim. scores.
        """

        def traverse_tree(node, scores_dict):
            if node.comment_id not in scores_dict:
                scores_dict[node.comment_id] = {"scores": [], "parent_id": node.parent_comment_id}
            scores_dict[node.comment_id]["scores"].append(node.similarity_score)
            for child in node.children:
                traverse_tree(child, scores_dict)

        scores_dict = defaultdict(lambda: {"scores": [], "parent_id": None})
        
        for tree in trees:
            traverse_tree(tree, scores_dict)

        avg_scores = {}
        similarity_data = {
            "Comment ID": [], 
            "Parent ID": [], 
            "Similarity Scores": [], 
            "Average Similarity": [],
            "Standard Deviation": []
        }

        for comment_id, data in scores_dict.items():
            avg_score = sum(data["scores"]) / len(data["scores"])
            std_dev = np.std(data["scores"], ddof=0)  # Use ddof=0 for population standard deviation
            avg_scores[comment_id] = avg_score
            
            similarity_data["Comment ID"].append(comment_id)
            similarity_data["Parent ID"].append(data["parent_id"])
            similarity_data["Similarity Scores"].append(data["scores"])
            similarity_data["Average Similarity"].append(avg_score)
            similarity_data["Standard Deviation"].append(std_dev)

        similarity_df = pd.DataFrame(similarity_data)

        nodes = {}
        root_candidates = []

        for comment_id, avg_score in avg_scores.items():
            parent_id = scores_dict[comment_id]["parent_id"]
            node = SimilarityNode(comment_id, parent_id, avg_score)
            nodes[comment_id] = node

            if parent_id is None or parent_id == "None":
                root_candidates.append(node)
            elif parent_id in nodes:
                nodes[parent_id].add_child(node)

        root = root_candidates[0] if root_candidates else None

        return root, similarity_df

    def save_df(self, similarity_df, path):
        similarity_df.to_csv(path, index=False)

    def calculate_tree_layout(self, graph, root):
        """
        Calculate a tree-like layout for the graph manually.
        Nodes are placed in hierarchical layers based on depth.
        """
        levels = {}  # To store nodes by depth level
        visited = set()

        def assign_levels(node, depth=0):
            if node in visited:
                return
            visited.add(node)
            if depth not in levels:
                levels[depth] = []
            levels[depth].append(node)
            for neighbor in graph.successors(node):  # Traverse children
                assign_levels(neighbor, depth + 1)

        # Start from the root and assign levels
        assign_levels(root)

        # Assign positions based on levels
        pos = {}
        max_width = max(len(nodes) for nodes in levels.values())  # Max nodes in any level
        for depth, nodes in levels.items():
            x_positions = range(-max_width // 2, max_width // 2, max(1, max_width // len(nodes)))
            for i, node in enumerate(nodes):
                pos[node] = (x_positions[i], -depth)  # x: position in level, y: depth level
        return pos

    def plot_similarity_tree(self, tree, exp_name, post_id):
        """
        Takes a tree and plots it.
        """

        def build_graph(node, graph):
            graph.add_node(node.comment_id, similarity=node.similarity_score)
            for child in node.children:
                graph.add_edge(node.comment_id, child.comment_id)
                build_graph(child, graph)

        graph = nx.DiGraph()

        build_graph(tree, graph)

        # Get positions for nodes using a custom layout
        pos = self.calculate_tree_layout(graph, tree.comment_id)

        similarities = nx.get_node_attributes(graph, 'similarity')
        similarity_values = list(similarities.values())

        norm = mcolors.Normalize(vmin=min(similarity_values), vmax=max(similarity_values))
        cmap = plt.cm.Blues 
        node_colors = [cmap(norm(similarities[node])) for node in graph.nodes]

        fig, ax = plt.subplots(figsize=(12, 8))
        nx.draw(
            graph,
            pos,
            ax=ax,
            with_labels=True,
            node_size=1500,
            node_color=node_colors,
            font_size=8
        )

        plt.title(f"Tree with average similarity for post {post_id}, built using {exp_name}")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Similarity Score", rotation=270, labelpad=15)
        plt.show()

    def plot_avg_similarity(self, df, exp_name, post_id):
        """
        Plot the average similarity score per depth level. 
        """

        def calculate_depth(row, depth_dict):
            if row["Parent ID"] == "None":
                return 0
            if row["Parent ID"] in depth_dict:
                return depth_dict[row["Parent ID"]] + 1
            return 0

        depth_dict = {}
        
        for idx, row in df.iterrows():
            depth_dict[row["Comment ID"]] = calculate_depth(row, depth_dict)

        df["Depth"] = df["Comment ID"].map(depth_dict)

        df["Similarity Scores"] = df["Similarity Scores"].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        )

        exploded_df = df.explode("Similarity Scores")
        exploded_df["Similarity Scores"] = exploded_df["Similarity Scores"].astype(float)

        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=exploded_df,
            x="Depth",
            y="Similarity Scores",
            errorbar="sd", 
            marker="o"
        )
        
        plt.title(f"Avg similarity per depth for Post {post_id}, built using {exp_name}")
        plt.xlabel("Depth")
        plt.ylabel("Average Similarity Score")
        plt.grid(True)
        plt.show()

    def load_and_preprocess_data(self, directory, post_id, experiments):
        
        def calculate_depth(row, depth_dict):
            if row["Parent ID"] == "None":
                return 0
            if row["Parent ID"] in depth_dict:
                return depth_dict[row["Parent ID"]] + 1
            return 0

        dataframes = {}
        for exp in experiments:
            target_pattern = f"df_{exp}_{post_id}.csv"
            matching_files = [
                file for file in os.listdir(directory) if file == target_pattern
            ]
            if matching_files:
                file_path = os.path.join(directory, matching_files[0])
                df = pd.read_csv(file_path)

                # Calculate depths
                depth_dict = {}
                for idx, row in df.iterrows():
                    depth_dict[row["Comment ID"]] = calculate_depth(row, depth_dict)
                df["Depth"] = df["Comment ID"].map(depth_dict)

                # Process similarity scores
                df["Similarity Scores"] = df["Similarity Scores"].apply(
                    lambda x: eval(x) if isinstance(x, str) else x
                )
                exploded_df = df.explode("Similarity Scores")
                exploded_df["Similarity Scores"] = exploded_df["Similarity Scores"].astype(float)
                
                dataframes[exp] = exploded_df
            else:
                print(f"File for experiment '{exp}' and post ID '{post_id}' not found.")
                return None
        return dataframes

    def perform_tests(self, dataframes, output_file):
        experiments = list(dataframes.keys())

        with open(output_file, "w") as f:
            f.write("Statistical Test Results:\n\n")

            # Overall Multi-Group Comparisons
            avg_similarities = [dataframes[exp]["Average Similarity"] for exp in experiments]
            std_devs = [dataframes[exp]["Standard Deviation"] for exp in experiments]

            f.write("Overall Multi-Group Comparisons:\n")
            anova_sim_stat, anova_sim_p_value = f_oneway(*avg_similarities)
            kruskal_sim_stat, kruskal_sim_p_value = kruskal(*avg_similarities)
            f.write(f"  Average Similarity:\n")
            f.write(f"    ANOVA: statistic = {anova_sim_stat}, p-value = {anova_sim_p_value}\n")
            f.write(f"    Kruskal-Wallis: statistic = {kruskal_sim_stat}, p-value = {kruskal_sim_p_value}\n")

            anova_sd_stat, anova_sd_p_value = f_oneway(*std_devs)
            kruskal_sd_stat, kruskal_sd_p_value = kruskal(*std_devs)
            f.write(f"  Standard Deviation:\n")
            f.write(f"    ANOVA: statistic = {anova_sd_stat}, p-value = {anova_sd_p_value}\n")
            f.write(f"    Kruskal-Wallis: statistic = {kruskal_sd_stat}, p-value = {kruskal_sd_p_value}\n\n")

            # Overall Pairwise Comparisons
            f.write("Overall Pairwise Comparisons:\n")
            for i, exp1 in enumerate(experiments):
                for exp2 in experiments[i + 1:]:
                    sim1 = dataframes[exp1]["Average Similarity"]
                    sim2 = dataframes[exp2]["Average Similarity"]
                    sd1 = dataframes[exp1]["Standard Deviation"]
                    sd2 = dataframes[exp2]["Standard Deviation"]

                    # Paired t-tests and Wilcoxon tests
                    t_stat_sim, t_p_sim = ttest_rel(sim1, sim2)
                    w_stat_sim, w_p_sim = wilcoxon(sim1, sim2)
                    t_stat_sd, t_p_sd = ttest_rel(sd1, sd2)
                    w_stat_sd, w_p_sd = wilcoxon(sd1, sd2)

                    f.write(f"  {exp1} vs {exp2}:\n")
                    f.write(f"    Average Similarity:\n")
                    f.write(f"      Paired t-test: t-statistic = {t_stat_sim}, p-value = {t_p_sim}\n")
                    f.write(f"      Wilcoxon test: statistic = {w_stat_sim}, p-value = {w_p_sim}\n")
                    f.write(f"    Standard Deviation:\n")
                    f.write(f"      Paired t-test: t-statistic = {t_stat_sd}, p-value = {t_p_sd}\n")
                    f.write(f"      Wilcoxon test: statistic = {w_stat_sd}, p-value = {w_p_sd}\n\n")

            # Depth-Based Comparisons
            f.write("Depth-Based Comparisons:\n")
            depths = set.union(*[set(df["Depth"]) for df in dataframes.values()])
            for depth in sorted(depths):
                similarities_by_exp = {
                    exp: df[df["Depth"] == depth]["Similarity Scores"].dropna().tolist()
                    for exp, df in dataframes.items()
                }
                if any(len(similarities) == 0 for similarities in similarities_by_exp.values()):
                    continue

                f.write(f"Depth {depth}:\n")
                anova_stat, anova_p = f_oneway(*similarities_by_exp.values())
                kruskal_stat, kruskal_p = kruskal(*similarities_by_exp.values())
                f.write(f"  Multi-Group Tests:\n")
                f.write(f"    ANOVA: statistic = {anova_stat}, p-value = {anova_p}\n")
                f.write(f"    Kruskal-Wallis: statistic = {kruskal_stat}, p-value = {kruskal_p}\n")
                f.write(f"  Pairwise Comparisons:\n")
                exp_list = list(similarities_by_exp.keys())
                for i, exp1 in enumerate(exp_list):
                    for exp2 in exp_list[i + 1:]:
                        sim1 = similarities_by_exp[exp1]
                        sim2 = similarities_by_exp[exp2]
                        t_stat, t_p = ttest_rel(sim1, sim2)
                        w_stat, w_p = wilcoxon(sim1, sim2)
                        f.write(f"    {exp1} vs {exp2}:\n")
                        f.write(f"      Paired t-test: t-statistic = {t_stat}, p-value = {t_p}\n")
                        f.write(f"      Wilcoxon test: statistic = {w_stat}, p-value = {w_p}\n")
                f.write("\n")

        print(f"Results written to {output_file}")

    def run_combined_analysis(self, directory, post_id, experiments):
        dataframes = self.load_and_preprocess_data(directory, post_id, experiments)
        if dataframes:
            output_file = os.path.join(directory, f"combined_analysis_results_{post_id}.txt")
            self.perform_tests(dataframes, output_file)