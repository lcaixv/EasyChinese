from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def run_nmf_demo():
    # Manually created term-document matrix
    term_doc_matrix = np.array([[1, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0]])

    # Create feature names for each column
    feature_names = [f"word{i}" for i in range(term_doc_matrix.shape[0])]

    # Apply NMF
    nmf_model = NMF(n_components=3, init='random', random_state=0)
    W = nmf_model.fit_transform(term_doc_matrix.T)
    H = nmf_model.components_

    # Display the topics and their word distributions
    print("Topics and their word distributions:")
    for topic_idx, topic in enumerate(H):
        print(f"Topic {topic_idx}: {', '.join([feature_names[i] for i in topic.argsort()[:-6 - 1:-1]])}")

    # Displaying the topic associations for documents
    print("\nDocument associations to topics:")
    for doc_idx, doc in enumerate(W):
        print(f"Document {doc_idx}: Topic distribution {doc}")

if __name__ == "__main__":
    run_nmf_demo()
