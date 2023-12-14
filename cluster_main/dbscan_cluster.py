import pickle
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import cosine_distances

# Function to format numbers
def format_number(number):
    if isinstance(number, float):
        # Format with 3 decimal places
        return "{:.3f}".format(number).rstrip('0').rstrip('.')
    return number

# Load the embeddings and hanzi data from their respective files
with open('hanzi_embeddings_norm.pkl', 'rb') as f:
    embeddings = pickle.load(f)

with open('hanzi_list.pkl', 'rb') as f:
    hanzi_data = pickle.load(f)

# Create a DataFrame for the hanzi characters
hanzi_df = pd.DataFrame(hanzi_data, columns=['character'])

# Set DBSCAN parameters and perform DBSCAN clustering
eps_value = 0.1775  # example value, you'll need to tune this
min_samples_value = 2  # example value, you'll need to tune this
dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value, metric='cosine')
labels = dbscan.fit_predict(embeddings)
hanzi_df['Cluster'] = labels

# Compute the cosine distance matrix
distance_matrix = cosine_distances(embeddings)

# Compute silhouette scores for non-noise points
non_noise_indices = [i for i, label in enumerate(labels) if label != -1]
non_noise_labels = [label for label in labels if label != -1]
non_noise_embeddings = embeddings[non_noise_indices]  # Convert to NumPy array
if len(non_noise_indices) > 1:  # at least 2 points to compute silhouette score
    non_noise_distances = distance_matrix[:, non_noise_indices][non_noise_indices]
    cosine_silhouette_vals = silhouette_samples(non_noise_distances, non_noise_labels, metric='precomputed')
    hanzi_df.loc[hanzi_df['Cluster'] != -1, 'Cosine Silhouette'] = cosine_silhouette_vals
else:
    hanzi_df['Cosine Silhouette'] = 0  # Assign 0 silhouette for noise points or if not enough points for calculation

# Print DBSCAN clustering information
print(f"DBSCAN, eps: {eps_value}, min_samples: {min_samples_value}\n")
print(f"Number of clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")  # exclude noise if present
print(f"Number of noise points: {list(labels).count(-1)}\n")

# Iterate over each cluster sorted by silhouette score and print hanzi characters
# Note: DBSCAN may not form clusters for all data, so we'll only consider non-noise clusters here
non_noise_cluster_silhouette_scores = hanzi_df[hanzi_df['Cluster'] != -1].groupby('Cluster')['Cosine Silhouette'].mean().sort_values(ascending=False)
for cluster_id in non_noise_cluster_silhouette_scores.index:
    cluster_data = hanzi_df[(hanzi_df['Cluster'] == cluster_id) & (hanzi_df['Cluster'] != -1)].sort_values(by='Cosine Silhouette', ascending=False)
    cluster_characters = ''.join(cluster_data['character'].tolist())
    cluster_silhouette_score = format_number(non_noise_cluster_silhouette_scores[cluster_id])
    print(f"Cluster {cluster_id} Silhouette Score {cluster_silhouette_score}: {cluster_characters}")

# Get the descriptive statistics for non-noise character silhouette scores, cluster silhouette scores, and cluster sizes
character_silhouette_desc = hanzi_df[hanzi_df['Cluster'] != -1]['Cosine Silhouette'].describe()
cluster_silhouette_desc = non_noise_cluster_silhouette_scores.describe()
cluster_size_desc = hanzi_df[hanzi_df['Cluster'] != -1]['Cluster'].value_counts().describe()

# Combine the descriptive statistics into one DataFrame
stats_df = pd.DataFrame({
    'Character Silhouette': character_silhouette_desc,
    'Cluster Silhouette': cluster_silhouette_desc,
    'Cluster Size': cluster_size_desc
}).applymap(format_number).fillna(0)

# Print the statistical summary for the clustering
print("\nDBSCAN, Outliers Excluded")
print(stats_df)
