import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples

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

def format_number(number):
    if isinstance(number, float):
        # Format with 3 decimal places
        return "{:.3f}".format(number).rstrip('0').rstrip('.')
    return number

def main():
    similarity_file = 'pairwise_prob.pkl'
    list_file = 'hanzi_list.pkl'
    num_clusters = int(input("Enter the number of clusters: "))

    similarity_matrix, characters_list = load_data(similarity_file, list_file)
    labels = hierarchical_clustering(similarity_matrix, num_clusters)

    # Create a DataFrame with characters and their respective cluster labels
    hanzi_df = pd.DataFrame({'character': characters_list, 'Cluster': labels})

    # Convert similarity to dissimilarity for silhouette calculation
    dissimilarity_matrix = 1 - similarity_matrix

    # Calculate silhouette scores
    silhouette_vals = silhouette_samples(dissimilarity_matrix, labels, metric='precomputed')
    hanzi_df['Cosine Silhouette'] = silhouette_vals

    # Calculate descriptive statistics
    character_silhouette_desc = hanzi_df['Cosine Silhouette'].describe()
    cluster_silhouette_scores = hanzi_df.groupby('Cluster')['Cosine Silhouette'].mean()
    cluster_silhouette_desc = cluster_silhouette_scores.describe()
    cluster_size_desc = hanzi_df['Cluster'].value_counts().describe()

    # Combine the descriptive statistics into one DataFrame
    stats_df = pd.DataFrame({
        'Character Silhouette': character_silhouette_desc,
        'Cluster Silhouette': cluster_silhouette_desc,
        'Cluster Size': cluster_size_desc
    }).applymap(format_number).fillna(0)

    print("\nConsensus K-Means via Complete-Linkage, 214 Clusters")
    print(stats_df)

    # Sort the clusters by silhouette score in descending order and print
    print("\nClusters sorted by silhouette score:")
    for cluster_id in cluster_silhouette_scores.sort_values(ascending=False).index:
        cluster_data = hanzi_df[hanzi_df['Cluster'] == cluster_id].sort_values(by='Cosine Silhouette', ascending=False)
        cluster_characters = ''.join(cluster_data['character'].tolist())
        cluster_silhouette_score = format_number(cluster_silhouette_scores[cluster_id])
        print(f"Cluster {cluster_id} (Silhouette Score: {cluster_silhouette_score}): {cluster_characters}")

if __name__ == "__main__":
    main()
