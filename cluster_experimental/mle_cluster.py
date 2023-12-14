import numpy as np
import pickle

# Function to calculate log odds matrix
def calculate_log_odds_matrix(pairwise_prob_matrix):
    return np.log(pairwise_prob_matrix) - np.log(1 - pairwise_prob_matrix)

# Function to compute the total log likelihood for a given clustering
def compute_log_likelihood(clustering, log_odds_matrix, char_to_index_map):
    total_log_likelihood = 0

    # Iterate over each cluster
    for cluster in clustering:
        # Convert characters to indices using the mapping
        cluster_indices = [char_to_index_map[char] for char in cluster]

        # Calculate log likelihood for all pairs within the cluster
        for i in range(len(cluster_indices)):
            for j in range(i + 1, len(cluster_indices)):
                total_log_likelihood += log_odds_matrix[cluster_indices[i], cluster_indices[j]]

    return total_log_likelihood

# Load the pairwise probability matrix and the list of all Chinese characters
with open('pairwise_prob.pkl', 'rb') as file:
    pairwise_prob = pickle.load(file)

with open('hanzi_list.pkl', 'rb') as file:
    hanzi_list = pickle.load(file)

# Define your target characters
target_characters = ['一', '二', '日', '月', '猫', '狗']

# Create a dictionary to map characters to new indices (0 to 5)
char_to_index_map = {char: i for i, char in enumerate(target_characters)}

# Extract the submatrix for the target characters
indices = [hanzi_list.index(char) for char in target_characters]
pairwise_prob_submatrix = pairwise_prob[np.ix_(indices, indices)]

# Calculate the log odds matrix for the submatrix
log_likelihood_submatrix = np.log(pairwise_prob_submatrix)

# Define the clustering
clustering = [['一','二'],['日'],['月'],['猫'], ['狗']]

# Compute the total log likelihood of the clustering
total_likelihood = compute_log_likelihood(clustering, log_likelihood_submatrix, char_to_index_map)

print("Total Log Likelihood of the Clustering:", total_likelihood)
