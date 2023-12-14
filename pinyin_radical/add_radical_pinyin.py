import csv
from pypinyin import pinyin, Style

# Load hanziDB.csv to create a character to radical mapping
def load_radicals():
    with open("hanziDB.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["character"]: row["radical"] for row in reader}

# Strip tones from the pinyin string
def strip_tones(pinyin_with_tones):
    return ''.join([char for char in pinyin_with_tones if not char.isdigit()])

# Convert word to the required formats
def word_to_data(word, radicals_dict):
    pinyin_with_tone = pinyin(word)
    pinyin_without_tone = [strip_tones(py[0]) for py in pinyin(word, style=Style.TONE3)]

    radical_tone = []
    radical_toneless = []
    tone = []
    toneless = []

    for char in word:
        radical = radicals_dict.get(char, char)
        char_pinyin_tone = pinyin(char)[0][0]
        char_pinyin_toneless = strip_tones(pinyin(char, style=Style.TONE3)[0][0])

        radical_tone.append(f"{radical}{char_pinyin_tone}")
        radical_toneless.append(f"{radical}{char_pinyin_toneless}")
        tone.append(char_pinyin_tone)
        toneless.append(char_pinyin_toneless)

    return {
        "radical_tone": " ".join(radical_tone),
        "radical_toneless": " ".join(radical_toneless),
        "tone": " ".join(tone),
        "toneless": " ".join(toneless),
    }

def main():
    radicals_dict = load_radicals()

    with open("word_frequency.txt", "r", encoding="utf-8") as f, open("word_frequency_annotated.txt", "w", encoding="utf-8", newline='') as out:
        reader = csv.reader(f, delimiter="\t")
        writer = csv.writer(out, delimiter="\t")
        
        # Write header
        writer.writerow(["word", "frequency", "radical_tone", "radical_toneless", "tone", "toneless"])

        for row in reader:
            word, frequency = row
            data = word_to_data(word, radicals_dict)
            writer.writerow([word, frequency, data["radical_tone"], data["radical_toneless"], data["tone"], data["toneless"]])

if __name__ == "__main__":
    main()
