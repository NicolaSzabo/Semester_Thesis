import os
import pandas as pd

# Define the paths
full_body_path = 'home/fit_member/Documents/NS_SemesterWork/Project/data/niftis_full_body'
masked_with_nan_path = 'home/fit_member/Documents/NS_SemesterWork/Project/data/preprocessing/masked_with_nan'
excel_file_path = 'home/fit_member/Documents/NS_SemesterWork/Project/data/data_overview.xlsx'

# List files in each directory
full_body_files = set(os.listdir(full_body_path))  # Set for faster lookups
masked_files = set(os.listdir(masked_with_nan_path))

# Load the Excel file
data = pd.read_excel(excel_file_path)

# Check if the file from full_body_path exists in masked_with_nan_path
data['masked_CT'] = data['Filename'].apply(lambda x: 'yes' if x in masked_files else 'no')

# Save the updated Excel file
updated_excel_path = 'home/fit_member/Documents/NS_SemesterWork/Project/data/data_overview_updated.xlsx'
data.to_excel(updated_excel_path, index=False)

print(f"Updated Excel file saved at: {updated_excel_path}")
