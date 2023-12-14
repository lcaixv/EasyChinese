import pickle
import numpy as np
from sklearn.cluster import AgglomerativeClustering

def load_data(similarity_file, list_file):
    with open(similarity_file, 'rb') as f:
        similarity_matrix = pickle.load(f)
    with open(list_file, 'rb') as f:
        characters_list = pickle.load(f)
    return similarity_matrix, characters_list

def hierarchical_clustering(similarity_matrix, num_clusters):
    # Convert similarity to dissimilarity
    dissimilarity_matrix = 1 - similarity_matrix
    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity='precomputed', linkage='complete')
    clustering.fit(dissimilarity_matrix)
    return clustering.labels_

def main():
    similarity_file = 'pairwise_prob.pkl'
    list_file = 'hanzi_list.pkl'
    num_clusters = int(input("Enter the number of clusters: "))

    similarity_matrix, characters_list = load_data(similarity_file, list_file)
    labels = hierarchical_clustering(similarity_matrix, num_clusters)

    # Output the results
    clusters = {i: [] for i in range(num_clusters)}
    for i, label in enumerate(labels):
        clusters[label].append(characters_list[i])

    for cluster, characters in clusters.items():
        print(f"Cluster {cluster}: {''.join(characters)}")

if __name__ == "__main__":
    main()
