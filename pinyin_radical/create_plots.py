import matplotlib.pyplot as plt

def process_file(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()[1:]  # Skipping the header
        length_confidence = {}  # Dictionary to store sum of the most_popular_word_frequency and sum of total_frequency for each unique word length
        for line in lines:
            _, total_freq, word, word_freq, _ = line.strip().split('\t')
            length = len(word)
            if length >= 4:
                length = '4+'  # Grouping lengths of 4 or more
            else:
                length = str(length)  # Convert to string to maintain consistency
            if length not in length_confidence:
                length_confidence[length] = [0, 0]  # Initializing to [sum_word_freq, sum_total_freq]
            length_confidence[length][0] += int(word_freq)
            length_confidence[length][1] += int(total_freq)
    
    # Calculating confidence for each word length
    for length in length_confidence:
        length_confidence[length] = length_confidence[length][0] / length_confidence[length][1]
    
    return length_confidence

# Process the summary_tone file and get the confidence values
data = process_file('summary_radical_tone.txt')

# Plotting the data using a bar plot
plt.figure(figsize=(10, 6))
x = sorted(data.keys(), key=lambda k: (k == '4+', k))  # Sort keys so '4+' is last
y = [data[k] for k in x]
plt.bar(x, y, color='blue')

plt.title('Radical and Pinyin with Grouped Words: Confidence vs. Word Length')
plt.xlabel('Word Length')
plt.ylabel('Confidence')
plt.xticks(x)  # Set x-axis tick labels
plt.yticks([i * 0.1 for i in range(11)])  # Set y-axis ticks with 0.1 increments
plt.grid(axis='y')
plt.tight_layout()
plt.show()
