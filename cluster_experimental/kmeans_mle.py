import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load the embeddings and the hanzi list
with open('hanzi_embeddings_norm.pkl', 'rb') as file:
    embeddings = pickle.load(file)

with open('hanzi_list.pkl', 'rb') as file:
    hanzi_list = pickle.load(file)

assert len(hanzi_list) == len(embeddings), "Mismatch between list and embeddings length."

num_clusters = 214
num_iterations = 100

# Initialize matrix P
P = np.zeros((len(hanzi_list), len(hanzi_list)))
cluster_assignments = []

for iteration in range(num_iterations):
    kmeans = KMeans(n_clusters=num_clusters, n_init=1, random_state=iteration)
    clusters = kmeans.fit_predict(embeddings)
    cluster_assignments.append(clusters)
    for cluster in range(num_clusters):
        indices = np.where(clusters == cluster)[0]
        P[np.ix_(indices, indices)] += 1
    print(f"K-means run {iteration + 1} complete.")

# Normalize to get probabilities
P /= num_iterations

# Compute log probabilities for P and 1 - P
log_prob_P = np.log(P)
log_prob_not_P = np.log(1 - P)

log_odds = np.log(P) - np.log(1 - P)

# pickle log_odds as a file
with open('log_odds.pkl', 'wb') as file:
    pickle.dump(log_odds, file)

# Lists to store results
clustering_log_probs = []
silhouette_scores = []

# Compute the clustering log probability and silhouette score for each iteration
for clusters in cluster_assignments:
    cluster_mask = clusters[:, None] == clusters[None, :]
    clustering_log_prob = np.sum(np.where(cluster_mask, log_prob_P, log_prob_not_P))
    silhouette_avg = silhouette_score(embeddings, clusters)
    clustering_log_probs.append(clustering_log_prob)
    silhouette_scores.append(silhouette_avg)

# Plotting
plt.scatter(clustering_log_probs, silhouette_scores)
plt.xlabel('Clustering Log Probability')
plt.ylabel('Silhouette Score')
plt.title('Clustering Log Probability vs Silhouette Score')
plt.show()
