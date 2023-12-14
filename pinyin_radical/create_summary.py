import csv

def read_file(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)  # skip header
        for row in reader:
            data.append(row)
    return data

def generate_summary(data, column_index, output_filename):
    summary = {}
    
    for row in data:
        word = row[0]
        frequency = int(row[1])
        key = row[column_index]
        
        if key in summary:
            summary[key]["total_frequency"] += frequency
            if frequency > summary[key]["most_popular_word_frequency"]:
                summary[key]["most_popular_word"] = word
                summary[key]["most_popular_word_frequency"] = frequency
        else:
            summary[key] = {
                "total_frequency": frequency,
                "most_popular_word": word,
                "most_popular_word_frequency": frequency
            }
    
    with open(output_filename, 'w', encoding='utf-8') as file:
        file.write(f"pronunciation\ttotal_frequency\tmost_popular_word\tmost_popular_word_frequency\tconfidence\n")
        for key, value in sorted(summary.items(), key=lambda x: x[1]["total_frequency"], reverse=True):
            confidence = value["most_popular_word_frequency"] / value["total_frequency"]
            file.write(f"{key}\t{value['total_frequency']}\t{value['most_popular_word']}\t{value['most_popular_word_frequency']}\t{confidence:.3f}\n")

def main():
    data = read_file("word_frequency_annotated.txt")
    generate_summary(data, 2, "summary_radical_tone.txt")
    generate_summary(data, 3, "summary_radical_toneless.txt")
    generate_summary(data, 4, "summary_tone.txt")
    generate_summary(data, 5, "summary_toneless.txt")

if __name__ == "__main__":
    main()
