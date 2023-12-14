import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.metrics import silhouette_samples
from sklearn.feature_extraction.text import CountVectorizer

# Load the embeddings and the list of characters
with open('hsk_embeddings_norm.pkl', 'rb') as f:
    embeddings = pickle.load(f)

with open('hsk_list.pkl', 'rb') as f:
    hanzi_list = pickle.load(f)

# K-Means Clustering
num_clusters = 100
num_runs = 100
documents = []

for run in range(num_runs):
    print(f"Running K-means: Run {run + 1}/{num_runs}")
    kmeans = KMeans(n_clusters=num_clusters, n_init=1, random_state=run)
    kmeans.fit(embeddings)
    clusters = ['' for _ in range(num_clusters)]
    for idx, label in enumerate(kmeans.labels_):
        clusters[label] += hanzi_list[idx]
    documents.extend(clusters)

# Vectorize the documents
vectorizer = CountVectorizer(analyzer='char')
X = vectorizer.fit_transform(documents)

# Apply NMF
nmf = NMF(n_components=num_clusters, init='random', random_state=0)
W = nmf.fit_transform(X)
H = nmf.components_

# Topic distributions for each character
char_topic_distributions = H.T
char_indices = vectorizer.vocabulary_
inverse_char_indices = {v: k for k, v in char_indices.items()}

# Assign each character to the topic with the greatest topic probability
char_best_topics = np.argmax(char_topic_distributions, axis=1)

# Open a file to write the output
with open('kmeans_nmf_output.txt', 'w') as output_file:
    # Calculate the silhouette scores for each character
    silhouette_vals = silhouette_samples(embeddings, char_best_topics)

    # Create a DataFrame for easier analysis
    hanzi_df = pd.DataFrame({
        'Character': [inverse_char_indices[i] for i in range(len(char_best_topics))],
        'Assigned Topic': char_best_topics,
        'Silhouette Score': silhouette_vals
    })

    # Add top 3 topic distributions to the DataFrame
    top_topics = np.argsort(char_topic_distributions, axis=1)[:, -3:]  # Get the top 3 topics
    top_topic_vals = np.sort(char_topic_distributions, axis=1)[:, -3:]  # Get the top 3 topic values

    hanzi_df['Top Topic 1'] = top_topics[:, 2]
    hanzi_df['Top Topic 1 Distribution'] = top_topic_vals[:, 2]
    hanzi_df['Top Topic 2'] = top_topics[:, 1]
    hanzi_df['Top Topic 2 Distribution'] = top_topic_vals[:, 1]
    hanzi_df['Top Topic 3'] = top_topics[:, 0]
    hanzi_df['Top Topic 3 Distribution'] = top_topic_vals[:, 0]

    # Print each character with its assigned topic and top 3 topic distributions
    for index, row in hanzi_df.iterrows():
        character = row['Character']
        topic = row['Assigned Topic']
        print(f"Character: {character}, Assigned Topic: {topic}, ", end='', file=output_file)
        for i in range(1, 4):
            top_topic = row[f'Top Topic {i}']
            top_topic_val = row[f'Top Topic {i} Distribution']
            print(f"Top Topic {i}: {top_topic} ({top_topic_val:.3f}), ", end='', file=output_file)
        print(file=output_file)

    # Calculate the mean silhouette score for each topic
    topic_silhouette_scores = hanzi_df.groupby('Assigned Topic')['Silhouette Score'].mean().sort_values(ascending=False)

    # Print each topic sorted by silhouette score and its characters
    print("\nNMF-based Clustering Information\n", file=output_file)
    for topic_id in topic_silhouette_scores.index:
        topic_data = hanzi_df[hanzi_df['Assigned Topic'] == topic_id].sort_values(by='Silhouette Score', ascending=False)
        topic_characters = ''.join(topic_data['Character'].tolist())
        topic_silhouette_score = topic_silhouette_scores[topic_id]
        print(f"{topic_silhouette_score:.3f}: {topic_characters}", file=output_file)

    # Get the descriptive statistics
    character_silhouette_desc = hanzi_df['Silhouette Score'].describe()
    topic_silhouette_desc = topic_silhouette_scores.describe()
    topic_size_desc = hanzi_df['Assigned Topic'].value_counts().describe()

    # Combine the descriptive statistics into one DataFrame
    stats_df = pd.DataFrame({
        'Character Silhouette': character_silhouette_desc,
        'Topic Silhouette': topic_silhouette_desc,
        'Topic Size': topic_size_desc
    })

    print("\nDescriptive Statistics\n", file=output_file)
    print(stats_df, file=output_file)
