import pickle
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import cosine_distances

# Function to format numbers
def format_number(number):
    if isinstance(number, float):
        return "{:.3f}".format(number).rstrip('0').rstrip('.')
    return number

# Load the embeddings and hanzi data
with open('hanzi_embeddings_norm.pkl', 'rb') as f:
    embeddings = pickle.load(f)

with open('hanzi_list.pkl', 'rb') as f:
    hanzi_data = pickle.load(f)

# Create a DataFrame for the hanzi characters
hanzi_df = pd.DataFrame(hanzi_data, columns=['character'])

# Set number of clusters
n_clusters = 214

# Initialize a dictionary to store all statistics
statistics = {}

# Perform hierarchical clustering with Single linkage
single_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='single')
single_labels = single_clustering.fit_predict(embeddings)
hanzi_df['Single Cluster'] = single_labels

# Perform hierarchical clustering with Average linkage
average_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
average_labels = average_clustering.fit_predict(embeddings)
hanzi_df['Average Cluster'] = average_labels

# Perform hierarchical clustering with Complete linkage
complete_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
complete_labels = complete_clustering.fit_predict(embeddings)
hanzi_df['Complete Cluster'] = complete_labels

# Perform hierarchical clustering with Ward linkage
ward_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
ward_labels = ward_clustering.fit_predict(embeddings)
hanzi_df['Ward Cluster'] = ward_labels

# Compute the cosine distance matrix
distance_matrix = cosine_distances(embeddings)

# Compute silhouette scores for Single linkage
single_silhouette_vals = silhouette_samples(distance_matrix, single_labels, metric='precomputed')
hanzi_df['Single Silhouette'] = single_silhouette_vals

# Compute silhouette scores for Average linkage
average_silhouette_vals = silhouette_samples(distance_matrix, average_labels, metric='precomputed')
hanzi_df['Average Silhouette'] = average_silhouette_vals

# Compute silhouette scores for Complete linkage
complete_silhouette_vals = silhouette_samples(distance_matrix, complete_labels, metric='precomputed')
hanzi_df['Complete Silhouette'] = complete_silhouette_vals

# Compute silhouette scores for Ward linkage
ward_silhouette_vals = silhouette_samples(distance_matrix, ward_labels, metric='precomputed')
hanzi_df['Ward Silhouette'] = ward_silhouette_vals

# Sort the clusters by silhouette score in descending order for Single linkage
single_cluster_silhouette_scores = hanzi_df.groupby('Single Cluster')['Single Silhouette'].mean().sort_values(ascending=False)

# Sort the clusters by silhouette score in descending order for Average linkage
average_cluster_silhouette_scores = hanzi_df.groupby('Average Cluster')['Average Silhouette'].mean().sort_values(ascending=False)

# Sort the clusters by silhouette score in descending order for Complete linkage
complete_cluster_silhouette_scores = hanzi_df.groupby('Complete Cluster')['Complete Silhouette'].mean().sort_values(ascending=False)

# Sort the clusters by silhouette score in descending order for Ward linkage
ward_cluster_silhouette_scores = hanzi_df.groupby('Ward Cluster')['Ward Silhouette'].mean().sort_values(ascending=False)

# Collect statistics for all linkage methods
linkage_methods = ['Single', 'Average', 'Complete', 'Ward']

for method in linkage_methods:
    # Cluster Silhouette Scores
    cluster_silhouette_scores = hanzi_df.groupby(f'{method} Cluster')[f'{method} Silhouette'].mean().sort_values(ascending=False)
    
    # Descriptive Statistics
    character_silhouette_desc = hanzi_df[f'{method} Silhouette'].describe()
    cluster_silhouette_desc = cluster_silhouette_scores.describe()
    cluster_size_desc = hanzi_df[f'{method} Cluster'].value_counts().describe()
    
    statistics[method] = {
        'Cluster Silhouette Scores': cluster_silhouette_scores,
        'Descriptive Statistics': {
            'Character Silhouette': character_silhouette_desc,
            'Cluster Silhouette': cluster_silhouette_desc,
            'Cluster Size': cluster_size_desc
        }
    }

# Print all statistics together, including characters in each cluster
for method, stats in statistics.items():
    print(f"Hierarchical Clustering with {method} linkage, {n_clusters} Clusters\n")
    
    # Cluster Silhouette Scores and Characters
    cluster_silhouette_scores = stats['Cluster Silhouette Scores']
    for cluster_id in cluster_silhouette_scores.index:
        cluster_data = hanzi_df[hanzi_df[f'{method} Cluster'] == cluster_id].sort_values(by=f'{method} Silhouette', ascending=False)
        cluster_characters = ''.join(cluster_data['character'].tolist())
        cluster_silhouette_score = format_number(cluster_silhouette_scores[cluster_id])
        print(f"Cluster {cluster_id}, Silhouette Score {cluster_silhouette_score}: {cluster_characters}")
    
    # Descriptive Statistics
    stats_df = pd.DataFrame(stats['Descriptive Statistics']).applymap(format_number).fillna('')
    
    print(f"\n{method}-Linkage Clustering, {n_clusters} Clusters")
    print(stats_df)
    print("\n" + "="*50 + "\n")
