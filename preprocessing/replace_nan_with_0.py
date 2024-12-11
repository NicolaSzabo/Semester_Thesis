import os
import pandas as pd
import nibabel as nib
import numpy as np


def replace_nan_in_nifti(input_file, output_file):
    """
    Replace NaN values in a NIfTI file with zeros and save the corrected file to the specified output path.

    Parameters:
    - input_file (str): Path to the input NIfTI file.
    - output_file (str): Path to save the corrected NIfTI file.
    """
    # Load the NIfTI file
    img = nib.load(input_file)
    img_data = img.get_fdata()

    # Replace NaN values with 0
    img_data = np.nan_to_num(img_data, nan=0)

    # Save the corrected NIfTI file
    corrected_img = nib.Nifti1Image(img_data, img.affine, img.header)
    nib.save(corrected_img, output_file)
    print(f"Processed: {input_file} -> {output_file}")


def process_files_from_excel(excel_path, input_folder, output_folder):
    """
    Process NIfTI files listed in an Excel file (column 'Nr') if they are missing in the output folder.

    Parameters:
    - excel_path (str): Path to the Excel file containing the file names in column 'Nr'.
    - input_folder (str): Path to the folder containing the original NIfTI files.
    - output_folder (str): Path to the folder where processed files will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Read the Excel file
    df = pd.read_excel(excel_path)

    # Iterate through the 'Nr' column
    for file_nr in df['Nr']:
        # Construct the input and output file paths
        input_file = os.path.join(input_folder, f"{file_nr}.nii.gz")
        output_file = os.path.join(output_folder, f"{file_nr}.nii.gz")

        # Check if the output file is missing
        if not os.path.exists(output_file):
            if os.path.exists(input_file):
                # Process the file if the input file exists
                replace_nan_in_nifti(input_file, output_file)
            else:
                print(f"Input file not found: {input_file}")
        else:
            print(f"Output file already exists: {output_file}")


# Paths
excel_path = 'G://data/data_overview_binary_cleaned.xlsx'
input_folder = "G://data//preprocessing//windowed_normalized_without_aorta"
output_folder = "G://data_final_without_aorta"

# Process files
process_files_from_excel(excel_path, input_folder, output_folder)
