import pickle
import numpy as np

# Load the original files
with open('hanzi_list.pkl', 'rb') as file:
    hanzi_list = pickle.load(file)

with open('hanzi_embeddings_norm.pkl', 'rb') as file:
    hanzi_embeddings = pickle.load(file)

# List of desired characters
desired_characters = ['猫', '狮', '虎', '狗', '狼', '狐', '鸡', '鸭', '鹰']

# Find indices of the desired characters
indices = [hanzi_list.index(char) for char in desired_characters]

# Extract corresponding embeddings
mini_embeddings = hanzi_embeddings[indices]

# Ensure it's a numpy array
if not isinstance(mini_embeddings, np.ndarray):
    mini_embeddings = np.array(mini_embeddings)

# Save the mini list
with open('animal_list.pkl', 'wb') as file:
    pickle.dump(desired_characters, file)

# Save the mini embeddings
with open('animal_embeddings_norm.pkl', 'wb') as file:
    pickle.dump(mini_embeddings, file)

# create cosine distance matrix from embeddings
from sklearn.metrics.pairwise import cosine_distances

cosine_distances = cosine_distances(mini_embeddings)
with open('animal_cos_dist.pkl', 'wb') as file:
    pickle.dump(cosine_distances, file)
