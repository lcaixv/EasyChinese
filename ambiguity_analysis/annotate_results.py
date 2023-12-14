from pypinyin import pinyin, Style

# Function to create the pinyin index dictionary from the first file
def create_pinyin_index_dict(file_path):
    pinyin_index_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    for line_number, line in enumerate(lines, start=1):
        for character in line.strip():
            # Generate Pinyin with tones
            character_pinyin = pinyin(character, style=Style.TONE, heteronym=False)
            pinyin_str = ''.join([item for sublist in character_pinyin for item in sublist])
            pinyin_index_dict[character] = pinyin_str + str(line_number)
    return pinyin_index_dict

# Function to process the word frequency file and add pinyin annotations
def annotate_word_frequency(file_path, pinyin_index_dict, output_file):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    with open(output_file, 'w', encoding='utf-8') as out_file:
        for line in lines:
            word, frequency = line.strip().split()
            # Annotate each character with pinyin + line number
            annotated_word = ''.join([pinyin_index_dict.get(char, char) for char in word])
            out_file.write(f'{word}\t{annotated_word}\t{frequency}\n')

# File paths
algo = 'ward'
first_file_path = f'{algo}_clusters.txt'  # Replace with your file path
word_frequency_file_path = 'word_frequency.txt' # Replace with your file path
output_file_path = f'{algo}_frequency.txt'       # Path for the output file

# Creating the pinyin index dictionary
pinyin_index_dict = create_pinyin_index_dict(first_file_path)

# Annotating the word frequency file
annotate_word_frequency(word_frequency_file_path, pinyin_index_dict, output_file_path)

print(f"Annotated word frequency file is saved as {output_file_path}")
