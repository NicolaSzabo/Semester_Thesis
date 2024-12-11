import os
import pandas as pd

# Path to the Excel file and NIfTI folder
excel_path = 'G://data//data_overview_binary_cleaned_256.xlsx'  # Replace with your actual Excel file path
nifti_folder = 'G://data_final_without_aorta'      # Replace with your actual folder path

# Load the Excel file
data = pd.read_excel(excel_path)

# List all NIfTI file names in the folder
nifti_files = [f.replace('.nii.gz', '') for f in os.listdir(nifti_folder) if f.endswith('.nii.gz')]

# Compare Excel column 'Nr' with the NIfTI file names
data['data_without_aorta'] = data['Nr'].apply(lambda x: 'yes' if str(x) in nifti_files else 'no')

# Save the updated DataFrame (optional)
output_path = 'G://data//data_overview_binary_cleaned_256.xlsx'
data.to_excel(output_path, index=False)

# Print the updated DataFrame
print(data)
