from gensim import corpora, models
import numpy as np

def run_lda_demo():
    # Manually created term-document matrix
    
    term_doc_matrix = np.array([[1, 0, 0, 1, 0, 0, 6, 0, 6, 0],
                                [1, 0, 0, 1, 0, 0, 6, 0, 6, 0],
                                [0, 1, 0, 0, 1, 0, 6, 0, 6, 0],
                                [0, 1, 0, 0, 1, 0, 0, 6, 0, 6],
                                [0, 0, 1, 0, 0, 1, 0, 6, 0, 6],
                                [0, 0, 1, 0, 0, 1, 0, 6, 0, 0.5]])
    
    term_doc_matrix *= 1

    # Convert the term-document matrix to a list of term frequency tuples for each document
    corpus = []
    for doc in term_doc_matrix.T:
        corpus.append(list(zip(range(len(doc)), doc)))

    # Creating a dictionary with mock "words" for each index
    dictionary = corpora.Dictionary([["word{}".format(i) for i in range(len(term_doc_matrix))]])

    # Applying the LDA model
    lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10000)

    # Display the topics and their distributions
    print("Topics and their word distributions:")
    for topic_id, topic in lda_model.print_topics(num_words=len(dictionary)):
        print(f"Topic {topic_id}: {topic}")

    # Displaying the topic associations of words
    print("\nTopic associations for each word:")
    for word_id, word in dictionary.iteritems():
        print(f"Word: {word}")
        for topic_num, prob in lda_model.get_term_topics(word_id):
            print(f"  Topic {topic_num}: Probability {prob}")

if __name__ == "__main__":
    run_lda_demo()
