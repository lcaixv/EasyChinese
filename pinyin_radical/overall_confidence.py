def calculate_ratio(filename):
    total_sum = 0
    popular_sum = 0

    with open(filename, 'r', encoding='utf-8') as file:
        next(file)  # Skip the header row
        for line in file:
            columns = line.strip().split('\t')
            if len(columns) >= 4:
                total_frequency = int(columns[1])
                most_popular_word = columns[2]
                most_popular_word_frequency = int(columns[3])

                if len(most_popular_word) == 1:
                    print(most_popular_word)

                total_sum += total_frequency * len(most_popular_word)
                popular_sum += most_popular_word_frequency * len(most_popular_word)

    if popular_sum != 0:
        return popular_sum / total_sum
    else:
        return "Division by zero error"

# Replace 'summary_tone.txt' with the path to your file
result = calculate_ratio('summary_radical_tone.txt')
print(result)
