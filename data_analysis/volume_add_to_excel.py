import os
import pandas as pd
import nibabel as nib
import numpy as np




def calculate_heart_volume(file_path):
    """
    Calculate the volume of a single NIfTI segmentation file.

    Parameters:
        file_path (str): Path to the NIfTI file.

    Returns:
        float: Volume in milliliters (mL), or None if the file doesn't exist.
    """
    if not os.path.exists(file_path):
        return None
    
    nifti_image = nib.load(file_path)
    nifti_data = nifti_image.get_fdata()
    header = nifti_image.header

    # Compute voxel size (mm³) and total volume
    voxel_dims = header.get_zooms()
    voxel_volume = voxel_dims[0] * voxel_dims[1] * voxel_dims[2]
    heart_voxel_count = (nifti_data > 0).sum()
    total_volume_mm3 = heart_voxel_count * voxel_volume
    return total_volume_mm3 / 1000  # Convert to mL





def add_volumes_to_excel(excel_path, data_folder, output_excel_path=None):
    """
    Add a 'Volume_mL' column to the Excel file with calculated volumes.

    Parameters:
        excel_path (str): Path to the input Excel file.
        data_folder (str): Folder containing the NIfTI files.
        output_excel_path (str): Path to save the updated Excel file. If None, overwrites the input file.
    """
    # Load the Excel file
    excel_data = pd.read_excel(excel_path)
    
    # Filter for 'good' quality rows
    good_quality_data = excel_data[excel_data['quality'] == 'good']

    # Initialize an empty volume column
    volumes = []

    # Calculate volumes for each 'good' quality row
    for _, row in good_quality_data.iterrows():
        patient_id = row['Nr']
        file_path = os.path.join(data_folder, f"{patient_id}.nii.gz")
        volume = calculate_heart_volume(file_path)
        volumes.append(volume if volume is not None else np.nan)

    # Add the volume column back to the full Excel data
    excel_data['Volume_mL'] = np.nan  # Initialize with NaN
    excel_data.loc[excel_data['quality'] == 'good', 'Volume_mL'] = volumes

    # Save the updated Excel file
    if output_excel_path is None:
        output_excel_path = excel_path  # Overwrite original file

    excel_data.to_excel(output_excel_path, index=False)
    print(f"Updated Excel file saved to: {output_excel_path}")

# Input paths
excel_path = "G://data//data_overview_binary_cleaned_256.xlsx"
data_folder = "G://data_final_without_aorta"
output_excel_path = "G://data//data_overview_binary_cleaned_256.xlsx"  # Optional output path

# Run the function
add_volumes_to_excel(excel_path, data_folder, output_excel_path)
