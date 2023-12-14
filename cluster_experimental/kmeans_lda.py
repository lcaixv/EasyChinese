import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import silhouette_score

with open('kmeans_lda_output.txt', 'w') as output_file:
    # Load the embeddings and the list of characters
    with open('hanzi_embeddings_norm.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    with open('hanzi_list.pkl', 'rb') as f:
        hanzi_list = pickle.load(f)

    # K-Means Clustering
    num_clusters_k_means = 214
    num_clusers_lda = 250
    num_runs = 100
    documents = []
    char_to_cluster = {}

    for run in range(num_runs):
        print(f"Running K-means: Run {run + 1}/{num_runs}")
        kmeans = KMeans(n_clusters=num_clusters_k_means, n_init=1, random_state=run)
        kmeans.fit(embeddings)

        clusters = ['' for _ in range(num_clusters_k_means)]
        for idx, label in enumerate(kmeans.labels_):
            clusters[label] += hanzi_list[idx]
            char_to_cluster[hanzi_list[idx]] = run * num_clusters_k_means + label

        documents.extend(clusters)

    # Create a document-term matrix
    unique_characters = set(''.join(hanzi_list))
    char_to_index = {char: i for i, char in enumerate(unique_characters)}
    doc_term_matrix = np.zeros((len(documents), len(unique_characters)))

    for doc_idx, doc in enumerate(documents):
        for char in doc:
            char_idx = char_to_index[char]
            doc_term_matrix[doc_idx, char_idx] += 1

    # Apply LDA
    lda = LatentDirichletAllocation(n_components=num_clusers_lda, random_state=0)
    lda.fit(doc_term_matrix)

    # Assign each character to a topic
    doc_topic_distributions = lda.transform(doc_term_matrix)
    char_topic_distributions = np.zeros((len(hanzi_list), num_clusers_lda))

    for idx, char in enumerate(hanzi_list):
        cluster_idx = char_to_cluster[char]
        char_topic_distributions[idx] = doc_topic_distributions[cluster_idx]

    char_best_topics = np.argmax(char_topic_distributions, axis=1)

    # Calculate the overall silhouette score
    overall_silhouette_score = silhouette_score(embeddings, char_best_topics)
    print(f"Overall Silhouette Score for LDA-based Clustering: {overall_silhouette_score}", file=output_file)

    # Output each character, its assigned topic, and top topic distributions
    for idx, char in enumerate(hanzi_list):
        assigned_topic = char_best_topics[idx]
        top_3_topics = sorted(enumerate(char_topic_distributions[idx]), key=lambda x: x[1], reverse=True)[:3]
        top_3_topics_formatted = [(topic_id, round(prob, 3)) for topic_id, prob in top_3_topics]
        print(f"{char} (Assigned to Topic {assigned_topic}): {top_3_topics_formatted}", file=output_file)

    # Print each topic and the characters assigned to it
    for topic_id in range(num_clusers_lda):
        topic_chars = [char for idx, char in enumerate(hanzi_list) if char_best_topics[idx] == topic_id]
        print(f"Topic {topic_id}: {' '.join(topic_chars)}", file=output_file)
