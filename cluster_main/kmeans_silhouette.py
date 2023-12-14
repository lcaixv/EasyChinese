import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the embeddings from the file
with open('hsk_embeddings_norm.pkl', 'rb') as f:
    embeddings = pickle.load(f)

# Load the DataFrame
with open('hsk_df.pkl', 'rb') as f:
    hsk_df = pickle.load(f)

# Perform k-means clustering (for example, using 50 clusters)
n_clusters = 100
kmeans = KMeans(n_clusters=n_clusters, n_init=100, random_state=0).fit(embeddings)

# Get the cluster labels for each entry (embedding)
labels = kmeans.labels_

# Create a new DataFrame that includes the cluster labels
labeled_df = hsk_df.copy()
labeled_df['Cluster'] = labels

# Group by cluster and print characters in each cluster
for cluster_id in range(n_clusters):
    cluster_characters = labeled_df[labeled_df['Cluster'] == cluster_id]['character'].tolist()
    print(f"Cluster {cluster_id}: {''.join(cluster_characters)}")

# # Determine k using the Silhouette method
# sil_scores = []  # Holds silhouette scores
# K = range(2, 150)  # Starts with 2 because silhouette score requires at least 2 clusters

# for k in K:
#     kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(embeddings)
#     silhouette_avg = silhouette_score(embeddings, kmeans.labels_)
#     sil_scores.append(silhouette_avg)
#     print(f'k = {k} completed. Silhouette score: {silhouette_avg}')

# # Plot the silhouette scores
# plt.figure(figsize=(10,6))
# plt.plot(K, sil_scores, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Silhouette Score')
# plt.title('The Silhouette Method showing the optimal k')
# plt.show()

# # If desired, you can find the optimal k programmatically
# optimal_k = K[np.argmax(sil_scores)]
# print(f'The optimal number of clusters is {optimal_k}')
