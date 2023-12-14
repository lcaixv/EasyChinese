# def calculate_character_frequencies(file_path):
#     # Dictionary to store the frequency of each character
#     char_freq = {}
    
#     # Read the file line by line
#     with open(file_path, 'r', encoding='UTF-8') as file:
#         for line in file:
#             # Strip whitespace and split by tab to separate the word and its frequency
#             parts = line.strip().split('\t')
#             if len(parts) == 2:
#                 word, freq = parts[0], int(parts[1])
#                 # Add the frequency to each character in the word
#                 for char in word:
#                     if char in char_freq:
#                         char_freq[char] += freq
#                     else:
#                         char_freq[char] = freq
    
#     return char_freq

# # The path to the file containing the word frequencies
# file_path = 'global_wordfreq.release_UTF-8.txt'  # Replace with the path to your file

# # Calculate the frequencies
# character_frequencies = calculate_character_frequencies(file_path)

# # The path to the output file
# output_file_path = 'character_frequencies.txt'  # Replace with your desired output path

# # Write the character frequencies to the output file, ignoring the BOM
# with open(output_file_path, 'w', encoding='UTF-8') as file:
#     # Write the character and frequency to the file, separated by a tab
#     for char, freq in character_frequencies.items():
#         if char != '\ufeff':  # Skip the BOM
#             file.write(f"{char}\t{freq}\n")

from collections import defaultdict

def extract_character_frequencies(input_file, output_file):
    char_freq = defaultdict(int)

    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                word, freq = parts
                freq = int(freq)
                for char in word:
                    char_freq[char] += freq
            else:
                print(f"Skipping malformed line: {line.strip()}")

    with open(output_file, 'w', encoding='utf-8') as file:
        for char, freq in sorted(char_freq.items()):
            file.write(f'{char}\t{freq}\n')

# Replace 'global_wordfreq.release_UTF-8.txt' with the path to your input file
# and 'character_frequencies.txt' with the desired output file path
extract_character_frequencies('global_wordfreq.release_UTF-8.txt', 'character_frequencies.txt')
