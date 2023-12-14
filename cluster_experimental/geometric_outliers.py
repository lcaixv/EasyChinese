import numpy as np
import pandas as pd
import pickle
from scipy.spatial.distance import pdist, squareform

# Function to calculate geometric ranked outlier score for a single character
def geometric_ranked_outlier_score(distances, r):
    # Sort distances based on the value of r
    sorted_distances = np.sort(distances[distances != 0])  # Exclude the zero distance to self

    if r > 1:
        sorted_distances = sorted_distances[::-1]
        r = 1 / r
    
    numerator = 0
    denominator = 0
    multiplier = 1

    # Calculate weighted sum and normalization sum using a for loop
    for i in range(len(sorted_distances)):
        numerator += sorted_distances[i] * multiplier
        denominator += multiplier
        multiplier *= r

    return numerator / denominator

# Function to compute scores for all characters
def compute_all_scores(dist_matrix, r):
    scores = []
    for distances in dist_matrix:
        scores.append(geometric_ranked_outlier_score(distances, r))
    return scores

# Load normalized word embeddings and hanzi list
with open('hanzi_embeddings_norm.pkl', 'rb') as file:  # Correct file path
    embeddings = pickle.load(file)

with open('hanzi_list.pkl', 'rb') as file:  # Correct file path
    hanzi_list = pickle.load(file)

# Compute pairwise cosine distances between embeddings
dist_matrix = squareform(pdist(embeddings, 'cosine'))

# Example usage
r = 0.5  # You can vary this parameter
outlier_scores = compute_all_scores(dist_matrix, r)

# Create a DataFrame to display hanzi and their outlier scores
outlier_scores_df = pd.DataFrame({
    'Hanzi': hanzi_list,
    'Outlier Score': outlier_scores
})

# Sort the DataFrame based on the outlier scores
outlier_scores_df.sort_values('Outlier Score', ascending=False, inplace=True)

# Updated Function to format and print the desired output
def print_formatted_scores(df, title):
    print(title)
    formatted_df = df.copy()
    formatted_df['Outlier Score'] = formatted_df['Outlier Score'].round(3)
    for index, row in formatted_df.iterrows():
        print(f"{row['Hanzi']} {row['Outlier Score']}")

# Print the value of r
print('Geometric Ranked Outlier Score\n')
print(f'r = {r}')

# Print the characters with the 20 highest outlier scores
print_formatted_scores(outlier_scores_df.head(20), "Hi-scores")

# Print the characters with the 20 lowest outlier scores, but in descending order of their scores
print_formatted_scores(outlier_scores_df.tail(20).iloc[::-1], "Lo-scores")
