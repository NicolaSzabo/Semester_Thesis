import pandas as pd
from collections import Counter

# Load the Excel file
file_path = "misclassified_files.xlsx"
df = pd.read_excel(file_path)

# Flatten all the columns into a single list
all_files = df.values.flatten()

# Remove any NaN values
all_files = [file for file in all_files if pd.notna(file)]

# Count occurrences of each filename
file_counts = Counter(all_files)

# Filter files that appear more than once
repeated_files = {file: count for file, count in file_counts.items() if count > 1}

# Sort the repeated files alphabetically by filename
sorted_repeated_files = sorted(repeated_files.items(), key=lambda x: x[0])

# Print sorted results
for file, count in sorted_repeated_files:
    print(f"{file} appears {count} times")

print(len(sorted_repeated_files))


