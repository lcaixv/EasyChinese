from pypinyin import pinyin, Style

# Function to get Pinyin for a character
def get_pinyin(character):
    # Use pinyin() function to get Pinyin for a list of characters
    pinyin_list = pinyin([character])
    # Join the list of Pinyin to form a string
    return ''.join(pinyin_list[0])

# Input and output file paths
input_file_path = 'character_freq.txt'
output_file_path = 'character_freq_with_pinyin.txt'

# Open the input file for reading and create the output file for writing
with open(input_file_path, 'r', encoding='utf-8') as input_file, \
     open(output_file_path, 'w', encoding='utf-8') as output_file:
    for line in input_file:
        # Split each line into character and frequency
        character, frequency = line.strip().split('\t')
        
        # Get the Pinyin for the character
        pinyin_result = get_pinyin(character)
        
        # Write the character, frequency, and Pinyin to the output file
        output_file.write(f"{character}\t{frequency}\t{pinyin_result}\n")

print(f"Pinyin data has been written to {output_file_path}")
