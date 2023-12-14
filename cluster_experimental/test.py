import pickle

def load_probability_matrix(file_name):
    with open(file_name, 'rb') as file:
        probability_matrix = pickle.load(file)
    return probability_matrix

def get_characters_indices(characters, characters_list):
    return [characters_list.index(char) for char in characters]

def extract_submatrix(matrix, indices):
    return matrix[indices][:, indices]

def print_matrix_with_labels(matrix, labels):
    # Print the top header
    print('\t' + '\t'.join(labels))
    # Print each row with the corresponding label and tab-separated values
    for label, row in zip(labels, matrix):
        row_str = '\t'.join([f"{int(elem)}" if elem in [0, 1] else f"{elem:.2f}" for elem in row])
        print(f"{label}\t{row_str}")

def main():
    probability_matrix_file = 'pairwise_prob.pkl'
    characters_file = 'hanzi_list.pkl'
    characters_to_extract = ['一', '二', '日', '月', '猫', '狗']

    # Load the probability matrix
    probability_matrix = load_probability_matrix(probability_matrix_file)

    # Load the list of characters
    with open(characters_file, 'rb') as file:
        characters_list = pickle.load(file)

    # Get indices of the specified characters
    indices = get_characters_indices(characters_to_extract, characters_list)

    # Extract the submatrix
    submatrix = extract_submatrix(probability_matrix, indices)

    # Print the submatrix with labels
    print("Same-cluster probabilities:")
    print_matrix_with_labels(submatrix, characters_to_extract)

if __name__ == "__main__":
    main()
