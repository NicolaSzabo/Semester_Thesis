import pandas as pd
import re

# Load the data
file_path = "/home/fit_member/Documents/NS_SemesterWork/Project/data/data_overview_binary_cleaned_256.xlsx" # Replace with your actual file path
data = pd.read_excel(file_path)

def extract_scan_year(nr):
    """
    Extract the scan year from the 'Nr' column.
    The scan year is represented as the first two digits of the 'Nr' + 2000.
    Example: '14-2321' -> 2014
    """
    match = re.match(r'(\d{2})-', nr)
    if match:
        return 2000 + int(match.group(1))  # Convert to full year
    return None

# Extract scan year and calculate age
data['Scan Year'] = data['Nr'].apply(extract_scan_year)
data['Age'] = data['Scan Year'] - data['Year of birth']

# Drop 'Scan Year' column if not needed
data = data.drop(columns=['Scan Year'])

# Save the updated data back to Excel
output_path = "/home/fit_member/Documents/NS_SemesterWork/Project/data/data_overview_binary_cleaned_256.xlsx"  # Save to the same file
data.to_excel(output_path, index=False)

print("Ages calculated based on scan year and saved successfully.")
