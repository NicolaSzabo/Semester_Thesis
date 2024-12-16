import os
import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

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
        dict: Average volumes and standard deviations per class.
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

    # Calculate averages and standard deviations
    avg_healthy_volume = np.mean(healthy_volumes) if healthy_volumes else 0
    std_healthy_volume = np.std(healthy_volumes) if healthy_volumes else 0
    avg_pathological_volume = np.mean(pathological_volumes) if pathological_volumes else 0
    std_pathological_volume = np.std(pathological_volumes) if pathological_volumes else 0

    # Print results
    print(f"Average Healthy Heart Volume: {avg_healthy_volume:.2f} mL (± {std_healthy_volume:.2f} mL)")
    print(f"Average Pathological Heart Volume: {avg_pathological_volume:.2f} mL (± {std_pathological_volume:.2f} mL)")

    # Improved Boxplot
    plt.figure(figsize=(8, 6))
    boxprops = dict(color='darkblue', linewidth=2)
    medianprops = dict(color='darkblue', linewidth=2)
    whiskerprops = dict(color='darkblue', linewidth=1.5)
    capprops = dict(color='darkblue', linewidth=2)

    plt.boxplot([healthy_volumes, pathological_volumes],
                labels=['Healthy', 'Pathological'],
                boxprops=boxprops, medianprops=medianprops,
                whiskerprops=whiskerprops, capprops=capprops, patch_artist=True,
                widths=0.6)

    plt.title('Heart Volume Comparison', fontsize=16)
    plt.ylabel('Volume (mL)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig('heart_volume_boxplot.png', dpi=300)
    print("Boxplot saved as 'heart_volume_boxplot.png' in the current directory.")

    return {
        "Healthy": {"Mean": avg_healthy_volume, "Std": std_healthy_volume},
        "Pathological": {"Mean": avg_pathological_volume, "Std": std_pathological_volume}
    }

# Input paths
excel_path = "/home/fit_member/Documents/NS_SemesterWork/Project/data/data_overview_binary_cleaned_256.xlsx"
data_folder = "/home/fit_member/Documents/NS_SemesterWork/Project/data_final_without_aorta"

# Run the function
average_volumes = calculate_average_volumes(excel_path, data_folder)
