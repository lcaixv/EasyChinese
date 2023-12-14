def process_line(line):
    parts = line.strip().split('\t')
    chinese_word, pinyin, frequency = parts[0], parts[1], int(parts[2])
    return chinese_word, pinyin, frequency

def main():
    data = {}

    with open('ward_frequency.txt', 'r', encoding='utf-8') as file:
        for line in file:
            chinese_word, pinyin, frequency = process_line(line)

            if pinyin not in data:
                data[pinyin] = {'total_freq': 0, 'most_popular': ('', 0)}
            
            data[pinyin]['total_freq'] += frequency

            if frequency > data[pinyin]['most_popular'][1]:
                data[pinyin]['most_popular'] = (chinese_word, frequency)

    with open('ward_summary.txt', 'w', encoding='utf-8') as out_file:
        for pinyin, info in data.items():
            out_file.write(f"{pinyin}\t{info['total_freq']}\t{info['most_popular'][0]}\t{info['most_popular'][1]}\n")

if __name__ == "__main__":
    main()
