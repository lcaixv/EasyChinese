import pickle
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import cosine_distances

# Function to format numbers
def format_number(number):
    if isinstance(number, float):
        # Format with 3 decimal places
        return "{:.3f}".format(number).rstrip('0').rstrip('.')
    return number

# Load the embeddings and hanzi data from their respective files
with open('hsk_embeddings_norm.pkl', 'rb') as f:
    embeddings = pickle.load(f)

with open('hsk_list.pkl', 'rb') as f:
    hanzi_data = pickle.load(f)

# Create a DataFrame for the hanzi characters
hanzi_df = pd.DataFrame(hanzi_data, columns=['character'])

# Set number of clusters to 214 and perform KMeans clustering
n_clusters = 100
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
labels = kmeans.fit_predict(embeddings)
hanzi_df['Cluster'] = labels

# Compute the cosine distance matrix and silhouette scores
distance_matrix = cosine_distances(embeddings)
cosine_silhouette_vals = silhouette_samples(distance_matrix, labels, metric='precomputed')
hanzi_df['Cosine Silhouette'] = cosine_silhouette_vals

# Sort the clusters by silhouette score in descending order
cluster_silhouette_scores = hanzi_df.groupby('Cluster')['Cosine Silhouette'].mean().sort_values(ascending=False)

# Print K-Means clustering information
print(f"K-Means, {n_clusters} Clusters\n")

# Iterate over each cluster sorted by silhouette score and print hanzi characters
for cluster_id in cluster_silhouette_scores.index:
    cluster_data = hanzi_df[hanzi_df['Cluster'] == cluster_id].sort_values(by='Cosine Silhouette', ascending=False)
    cluster_characters = ''.join(cluster_data['character'].tolist())
    cluster_silhouette_score = format_number(cluster_silhouette_scores[cluster_id])
    print(f"{cluster_silhouette_score}: {cluster_characters}")

# Get the descriptive statistics for character silhouette scores, cluster silhouette scores, and cluster sizes
character_silhouette_desc = hanzi_df['Cosine Silhouette'].describe()
cluster_silhouette_desc = cluster_silhouette_scores.describe()
cluster_size_desc = hanzi_df['Cluster'].value_counts().describe()

# Combine the descriptive statistics into one DataFrame
stats_df = pd.DataFrame({
    'Character Silhouette': character_silhouette_desc,
    'Cluster Silhouette': cluster_silhouette_desc,
    'Cluster Size': cluster_size_desc
}).applymap(format_number).fillna(0)

# Print the statistical summary for the clustering
print(f"\nK-Means, {n_clusters} Clusters")
print(stats_df)
