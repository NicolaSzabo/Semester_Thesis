import os
import pandas as pd
import nibabel as nib
import numpy as np

def calculate_intensity_metrics(file_path):
    """
    Calculate intensity metrics (mean, std, min, max) for a NIfTI segmentation.

    Parameters:
        file_path (str): Path to the NIfTI file.

    Returns:
        dict: Dictionary with mean, std, min, and max intensity.
    """
    if not os.path.exists(file_path):
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    
    nifti_image = nib.load(file_path)
    nifti_data = nifti_image.get_fdata()

    # Consider only non-zero voxels
    heart_voxels = nifti_data[nifti_data > 0]

    return {
        "mean": np.mean(heart_voxels),
        "std": np.std(heart_voxels),
        "min": np.min(heart_voxels),
        "max": np.max(heart_voxels),
    }

def add_intensity_metrics_to_excel(excel_path, data_folder, output_excel_path=None):
    """
    Add voxel intensity metrics to the Excel file.

    Parameters:
        excel_path (str): Path to the input Excel file.
        data_folder (str): Folder containing NIfTI files.
        output_excel_path (str): Path to save the updated Excel file. If None, overwrites the input file.
    """
    # Load and filter the Excel file for 'good' quality scans
    excel_data = pd.read_excel(excel_path)
    good_quality_data = excel_data[excel_data['quality'] == 'good']

    # Initialize new columns
    mean_intensities = []
    std_intensities = []
    min_intensities = []
    max_intensities = []

    # Calculate intensity metrics for each 'good' quality row
    for _, row in good_quality_data.iterrows():
        patient_id = row['Nr']
        file_path = os.path.join(data_folder, f"{patient_id}.nii.gz")
        metrics = calculate_intensity_metrics(file_path)
        
        mean_intensities.append(metrics["mean"])
        std_intensities.append(metrics["std"])
        min_intensities.append(metrics["min"])
        max_intensities.append(metrics["max"])

    # Add new columns back to the full Excel data
    excel_data['Mean_Intensity'] = np.nan
    excel_data['Std_Intensity'] = np.nan
    excel_data['Min_Intensity'] = np.nan
    excel_data['Max_Intensity'] = np.nan

    excel_data.loc[excel_data['quality'] == 'good', 'Mean_Intensity'] = mean_intensities
    excel_data.loc[excel_data['quality'] == 'good', 'Std_Intensity'] = std_intensities
    excel_data.loc[excel_data['quality'] == 'good', 'Min_Intensity'] = min_intensities
    excel_data.loc[excel_data['quality'] == 'good', 'Max_Intensity'] = max_intensities

    # Save the updated Excel file
    if output_excel_path is None:
        output_excel_path = excel_path  # Overwrite original file

    excel_data.to_excel(output_excel_path, index=False)
    print(f"Updated Excel file with intensity metrics saved to: {output_excel_path}")

# Input paths
excel_path = "G://data//data_overview_binary_cleaned_256.xlsx"
data_folder = "G://data_final_without_aorta"
output_excel_path = "G://data//data_overview_binary_cleaned_256.xlsx"

# Run the function
add_intensity_metrics_to_excel(excel_path, data_folder, output_excel_path)
