def calculate_ratios(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Initialize sums for different categories
    sums = {'1-char': [0, 0], '2-char': [0, 0], '3-char': [0, 0], '4+-char': [0, 0], 'weighted': [0, 0]}

    # Process each line
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) < 4:
            continue

        freq1 = int(parts[1])
        freq2 = int(parts[3])
        char_len = len(parts[2])

        # Classify based on the length of the Chinese character
        if char_len == 1:
            sums['1-char'][0] += freq1
            sums['1-char'][1] += freq2
        elif char_len == 2:
            sums['2-char'][0] += freq1
            sums['2-char'][1] += freq2
        elif char_len == 3:
            sums['3-char'][0] += freq1
            sums['3-char'][1] += freq2
        else:  # 4 or more characters
            sums['4+-char'][0] += freq1
            sums['4+-char'][1] += freq2

        # Weighted sums
        sums['weighted'][0] += freq1 * char_len
        sums['weighted'][1] += freq2 * char_len

    # Calculate and return the ratios
    return {key: val[1] / val[0] if val[0] != 0 else 0 for key, val in sums.items()}

# Replace 'gmm_summary.txt' with the path to your file
file_path = 'ward_summary.txt'
ratios = calculate_ratios(file_path)
print(ratios)
