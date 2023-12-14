import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
file_path = 'hanziDB.csv'
hanzi_db = pd.read_csv(file_path)

# Group the data by 'radical' and count the number of characters per radical
characters_per_radical = hanzi_db.groupby('radical')['character'].count()

# Calculate the mean, standard deviation, minimum, and maximum of characters per radical
mean_characters_per_radical = characters_per_radical.mean()
std_dev_characters_per_radical = characters_per_radical.std()
min_characters_per_radical = characters_per_radical.min()
max_characters_per_radical = characters_per_radical.max()

# Count the total number of unique radicals
num_radicals = hanzi_db['radical'].nunique()

# Create a box and whisker plot for the number of characters per radical
plt.figure(figsize=(10, 6))
plt.boxplot(characters_per_radical, vert=False)
plt.title('Box and Whisker Plot of Characters per Radical')
plt.xlabel('Number of Characters')
plt.yticks([1], ['Radicals'])
plt.show()

# Print the calculated statistics
print(f"Mean characters per radical: {mean_characters_per_radical}")
print(f"Standard deviation of characters per radical: {std_dev_characters_per_radical}")
print(f"Minimum number of characters per radical: {min_characters_per_radical}")
print(f"Maximum number of characters per radical: {max_characters_per_radical}")
print(f"Total number of unique radicals: {num_radicals}")
