import numpy as np
import pandas as pd
import pickle
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# Function to calculate geometric ranked outlier score for a single character
def geometric_ranked_outlier_score(distances, r):
    sorted_distances = np.sort(distances[distances != 0])
    if r > 1:
        sorted_distances = sorted_distances[::-1]
        r = 1 / r
    numerator = sum(d * (r ** i) for i, d in enumerate(sorted_distances))
    denominator = sum(r ** i for i in range(len(sorted_distances)))
    return numerator / denominator

# Function to compute scores for all characters
def compute_all_scores(dist_matrix, r):
    return [geometric_ranked_outlier_score(distances, r) for distances in dist_matrix]

# Load normalized word embeddings and hanzi list
with open('hsk_embeddings_norm.pkl', 'rb') as file:
    embeddings = pickle.load(file)
with open('hsk_list.pkl', 'rb') as file:
    hanzi_list = pickle.load(file)

# Compute pairwise cosine distances between embeddings
dist_matrix = squareform(pdist(embeddings, 'cosine'))

# Compute and assign outlier scores
r = 0.5
outlier_scores = compute_all_scores(dist_matrix, r)
outlier_scores_df = pd.DataFrame({'Hanzi': hanzi_list, 'Outlier Score': outlier_scores})

# Identify core and outlier points based on a threshold
outlier_threshold = 0.4
core_indices = outlier_scores_df[outlier_scores_df['Outlier Score'] < outlier_threshold].index
outlier_indices = outlier_scores_df[outlier_scores_df['Outlier Score'] >= outlier_threshold].index

# Perform K-means clustering on core points
n_clusters = 100
kmeans = KMeans(n_clusters=n_clusters, n_init=100, random_state=0)
core_embeddings = embeddings[core_indices]
kmeans.fit(core_embeddings)

# Assign outliers to the closest cluster
outlier_embeddings = embeddings[outlier_indices]
closest_clusters, _ = pairwise_distances_argmin_min(outlier_embeddings, kmeans.cluster_centers_)

# Organizing and printing the cluster assignments
cluster_assignments = {i: [] for i in range(n_clusters)}
for idx, cluster_idx in enumerate(kmeans.labels_):
    hanzi = hanzi_list[core_indices[idx]]
    cluster_assignments[cluster_idx].append(hanzi)

for idx, cluster_idx in enumerate(closest_clusters):
    hanzi = hanzi_list[outlier_indices[idx]]
    cluster_assignments[cluster_idx].append(f"{hanzi}*")  # Asterisk for outliers

for cluster_idx, members in cluster_assignments.items():
    print(f"Cluster {cluster_idx}: {' '.join(members)}")
