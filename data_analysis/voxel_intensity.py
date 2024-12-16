import os
import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def calculate_mean_intensity(file_path):
    """
    Calculate the mean voxel intensity of a single NIfTI segmentation file.

    Parameters:
        file_path (str): Path to the NIfTI file.

    Returns:
        float: Mean voxel intensity.
    """
    nifti_image = nib.load(file_path)
    nifti_data = nifti_image.get_fdata()

    # Compute mean intensity for non-zero voxels (inside the segmentation)
    mean_intensity = nifti_data[nifti_data > 0].mean()
    return mean_intensity

def calculate_average_intensities(excel_path, data_folder):
    """
    Calculate mean voxel intensities for 'healthy' and 'pathological' classes.

    Parameters:
        excel_path (str): Path to the Excel file.
        data_folder (str): Folder containing NIfTI files.

    Returns:
        dict: Average intensities and standard deviations per class.
    """
    # Load and filter the Excel file for 'good' quality scans
    excel_data = pd.read_excel(excel_path)
    good_quality_data = excel_data[excel_data['quality'] == 'good']

    # Initialize lists for mean intensities
    healthy_intensities = []
    pathological_intensities = []

    # Iterate over rows to calculate mean intensities
    for _, row in good_quality_data.iterrows():
        patient_id = row['Nr']
        classification = row['Classification']
        file_path = os.path.join(data_folder, f"{patient_id}.nii.gz")

        if os.path.exists(file_path):
            mean_intensity = calculate_mean_intensity(file_path)
            if classification == 'healthy':
                healthy_intensities.append(mean_intensity)
            elif classification == 'pathological':
                pathological_intensities.append(mean_intensity)
        else:
            print(f"File not found: {file_path}")

    # Calculate averages and standard deviations
    avg_healthy_intensity = np.mean(healthy_intensities) if healthy_intensities else 0
    std_healthy_intensity = np.std(healthy_intensities) if healthy_intensities else 0
    avg_pathological_intensity = np.mean(pathological_intensities) if pathological_intensities else 0
    std_pathological_intensity = np.std(pathological_intensities) if pathological_intensities else 0

    # Print results
    print(f"Average Healthy Mean Intensity: {avg_healthy_intensity:.2f} (± {std_healthy_intensity:.2f})")
    print(f"Average Pathological Mean Intensity: {avg_pathological_intensity:.2f} (± {std_pathological_intensity:.2f})")

    # Improved Boxplot
    plt.figure(figsize=(8, 6))
    boxprops = dict(color='darkblue', linewidth=2)
    medianprops = dict(color='darkblue', linewidth=2)
    whiskerprops = dict(color='darkblue', linewidth=1.5)
    capprops = dict(color='darkblue', linewidth=2)

    plt.boxplot([healthy_intensities, pathological_intensities],
                labels=['Healthy', 'Pathological'],
                boxprops=boxprops, medianprops=medianprops,
                whiskerprops=whiskerprops, capprops=capprops, patch_artist=True,
                widths=0.6)

    plt.title('Mean Voxel Intensity Comparison', fontsize=16)
    plt.ylabel('Mean Intensity (HU)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig('mean_intensity_boxplot.png', dpi=300)
    print("Boxplot saved as 'mean_intensity_boxplot.png' in the current directory.")

    return {
        "Healthy": {"Mean": avg_healthy_intensity, "Std": std_healthy_intensity},
        "Pathological": {"Mean": avg_pathological_intensity, "Std": std_pathological_intensity}
    }

# Input paths
excel_path = "/home/fit_member/Documents/NS_SemesterWork/Project/data/data_overview_binary_cleaned_256.xlsx"
data_folder = "/home/fit_member/Documents/NS_SemesterWork/Project/data_final_without_aorta"

# Run the function
average_intensities = calculate_average_intensities(excel_path, data_folder)
