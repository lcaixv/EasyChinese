from collections import defaultdict

# Initialize dictionaries to store the frequency sum, most common character, and its frequency for each pronunciation
pronunciation_frequency = defaultdict(int)
pronunciation_most_common = defaultdict(lambda: defaultdict(int))

# Read and process the file
with open("hanzi_freq.txt", "r", encoding="utf-8") as file:
    for line in file:
        columns = line.strip().split('\t')
        if len(columns) == 3:
            character, frequency, pronunciation = columns
            frequency = int(frequency)

            # Skip rows where the first column equals the third column
            if character != pronunciation:
                # Update the frequency sum for the pronunciation
                pronunciation_frequency[pronunciation] += frequency

                # Update the count and frequency of characters for the pronunciation
                character_counts = pronunciation_most_common[pronunciation]
                character_counts[character] += frequency

# Find the most common character and its frequency for each pronunciation
most_common_characters = {}
most_common_character_frequencies = {}
for pronunciation, character_counts in pronunciation_most_common.items():
    most_common_character = max(character_counts, key=character_counts.get)
    most_common_characters[pronunciation] = most_common_character
    most_common_character_frequencies[pronunciation] = character_counts[most_common_character]

# Calculate the sum of all most common character frequencies
sum_most_common_character_frequencies = sum(most_common_character_frequencies.values())

# Calculate the sum of all total frequencies
sum_total_frequencies = sum(pronunciation_frequency.values())

# Calculate the ratio
ratio = sum_most_common_character_frequencies / sum_total_frequencies if sum_total_frequencies != 0 else 0

# Print the ratio
print(f"Sum of All Most Common Character Frequencies / Sum of All Total Frequencies: {ratio}")
