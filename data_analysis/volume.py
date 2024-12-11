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
        float: Volume in milliliters (mL).
    """
    nifti_image = nib.load(file_path)
    nifti_data = nifti_image.get_fdata()
    header = nifti_image.header

    # Compute voxel size (mm³) and total volume
    voxel_dims = header.get_zooms()
    voxel_volume = voxel_dims[0] * voxel_dims[1] * voxel_dims[2]
    heart_voxel_count = (nifti_data > 0).sum()
    total_volume_mm3 = heart_voxel_count * voxel_volume
    return total_volume_mm3 / 1000  # Convert to mL

def calculate_average_volumes(excel_path, data_folder):
    """
    Calculate heart volumes for 'healthy' and 'pathological' classes.

    Parameters:
        excel_path (str): Path to the Excel file.
        data_folder (str): Folder containing NIfTI files.

    Returns:
        dict: Average volumes per class.
    """
    # Load and filter the Excel file for 'good' quality scans
    excel_data = pd.read_excel(excel_path)
    good_quality_data = excel_data[excel_data['quality'] == 'good']

    # Initialize lists for volumes
    healthy_volumes = []
    pathological_volumes = []

    # Iterate over rows to calculate volumes
    for _, row in good_quality_data.iterrows():
        patient_id = row['Nr']
        classification = row['Classification']
        file_path = os.path.join(data_folder, f"{patient_id}.nii.gz")

        if os.path.exists(file_path):
            volume = calculate_heart_volume(file_path)
            if classification == 'healthy':
                healthy_volumes.append(volume)
            elif classification == 'pathological':
                pathological_volumes.append(volume)
        else:
            print(f"File not found: {file_path}")

    # Calculate averages
    avg_healthy_volume = np.mean(healthy_volumes) if healthy_volumes else 0
    avg_pathological_volume = np.mean(pathological_volumes) if pathological_volumes else 0

    # Print results
    print(f"Average Healthy Heart Volume: {avg_healthy_volume:.2f} mL")
    print(f"Average Pathological Heart Volume: {avg_pathological_volume:.2f} mL")

    return {
        "Healthy": avg_healthy_volume,
        "Pathological": avg_pathological_volume
    }

# Input paths
excel_path = "G://data//data_overview_binary_cleaned_256.xlsx"
data_folder = "G://data_final_without_aorta"

# Run the function
average_volumes = calculate_average_volumes(excel_path, data_folder)

