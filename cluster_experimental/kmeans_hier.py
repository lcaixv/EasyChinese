import numpy as np
import pickle
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict

# Function to load data from a pickle file
def load_pickle(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)

# Load the data
similarity_matrix = load_pickle('log_odds.pkl')
labels = load_pickle('hanzi_list.pkl')

# Convert similarity to negative distance
distance_matrix = -similarity_matrix

# Replace -infinity with -1000000 and infinity with 1000000
negative_large_value = -1000000
positive_large_value = 1000000
distance_matrix[distance_matrix == -np.inf] = negative_large_value
distance_matrix[distance_matrix == np.inf] = positive_large_value

# Perform hierarchical clustering
linked = linkage(distance_matrix, method='average')

# Specify the number of clusters
num_clusters = 214  # Change this to your desired number of clusters

# Form flat clusters
clusters = fcluster(linked, num_clusters, criterion='maxclust')

# Group characters by cluster
clustered_characters = defaultdict(list)
for label, cluster_id in zip(labels, clusters):
    clustered_characters[cluster_id].append(label)

# Print characters in each cluster
for cluster_id, characters in clustered_characters.items():
    print(f"Cluster {cluster_id}: {', '.join(characters)}")
