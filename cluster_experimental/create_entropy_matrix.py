import pickle
import numpy as np

def shannon_entropy(p):
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

# Step 1: Load your data
with open('hsk_list.pkl', 'rb') as file:
    hanzi_list = pickle.load(file)

with open('pairwise_prob_hsk.pkl', 'rb') as file:
    pairwise_prob = pickle.load(file)

# Step 2 & 3: Compute Shannon Entropy and create new matrix
entropy_matrix = np.zeros_like(pairwise_prob)
for i in range(pairwise_prob.shape[0]):
    for j in range(pairwise_prob.shape[1]):
        entropy_matrix[i, j] = shannon_entropy(pairwise_prob[i, j])

# Step 4: Save or use the new matrix
# For example, saving it
with open('pairwise_entropy_hsk.pkl', 'wb') as file:
    pickle.dump(entropy_matrix, file)

# Or use it directly in your application
