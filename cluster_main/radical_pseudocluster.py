import pickle
import pandas as pd
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import cosine_distances

# Function to format numbers
def format_number(number):
    if isinstance(number, float):
        return "{:.3f}".format(number).rstrip('0').rstrip('.')
    return number

# Load the embeddings and hanzi data from their respective files
with open('hanzi_embeddings_norm.pkl', 'rb') as f:
    embeddings = pickle.load(f)

with open('hanzi_list.pkl', 'rb') as f:
    hanzi_list = pickle.load(f)

# Load the hanziDB.csv file
hanzi_db = pd.read_csv('hanziDB.csv')

hanzi_db = hanzi_db.dropna(subset=['radical', 'character'])


# Filter out characters that are not in the embeddings list
hanzi_db = hanzi_db[hanzi_db['character'].isin(hanzi_list)]

# Get embeddings for characters in the hanzi_db
embeddings_filtered = [embeddings[hanzi_list.index(char)] for char in hanzi_db['character']]

# Calculate the cosine distance matrix for the filtered embeddings
distance_matrix = cosine_distances(embeddings_filtered)

# Calculate silhouette scores
cosine_silhouette_vals = silhouette_samples(distance_matrix, hanzi_db['radical'], metric='precomputed')
hanzi_db['Cosine Silhouette'] = cosine_silhouette_vals

# Group by radical and calculate mean silhouette score
radical_silhouette_scores = hanzi_db.groupby('radical')['Cosine Silhouette'].mean().sort_values(ascending=False)

# Print radical-based clustering information
print("Radical-based Clustering\n")

for radical in radical_silhouette_scores.index:
    radical_data = hanzi_db[hanzi_db['radical'] == radical].sort_values(by='Cosine Silhouette', ascending=False)
    radical_characters = ''.join(radical_data['character'].tolist())
    radical_silhouette_score = format_number(radical_silhouette_scores[radical])
    print(f"{radical_silhouette_score}: {radical_characters}")

# Get the descriptive statistics for character silhouette scores, radical silhouette scores, and radical sizes
character_silhouette_desc = hanzi_db['Cosine Silhouette'].describe()
radical_silhouette_desc = radical_silhouette_scores.describe()
radical_size_desc = hanzi_db['radical'].value_counts().describe()

# Combine the descriptive statistics into one DataFrame
stats_df = pd.DataFrame({
    'Character Silhouette': character_silhouette_desc,
    'Radical Silhouette': radical_silhouette_desc,
    'Radical Size': radical_size_desc
}).applymap(format_number).fillna(0)

# Print the statistical summary for the radical-based clustering
print("\nKangxi Radicals")
print(stats_df)
