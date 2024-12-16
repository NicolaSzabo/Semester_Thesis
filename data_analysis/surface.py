import os
import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

def calculate_heart_surface(file_path):
    """
    Calculate the surface area of a single NIfTI segmentation file.

    Parameters:
        file_path (str): Path to the NIfTI file.

    Returns:
        float: Surface area in square millimeters (mm²).
    """
    nifti_image = nib.load(file_path)
    nifti_data = nifti_image.get_fdata()
    header = nifti_image.header

    # Compute voxel size (mm³)
    voxel_dims = header.get_zooms()
    voxel_surface_area = 2 * (voxel_dims[0] * voxel_dims[1] +
                              voxel_dims[1] * voxel_dims[2] +
                              voxel_dims[0] * voxel_dims[2])

    # Identify surface voxels using erosion
    binary_mask = nifti_data > 0
    eroded_mask = binary_erosion(binary_mask)
    surface_voxels = binary_mask & ~eroded_mask
    surface_voxel_count = np.sum(surface_voxels)

    # Total surface area
    total_surface_area_mm2 = surface_voxel_count * voxel_surface_area
    return total_surface_area_mm2

def calculate_average_surfaces(excel_path, data_folder):
    """
    Calculate heart surface areas for 'healthy' and 'pathological' classes.

    Parameters:
        excel_path (str): Path to the Excel file.
        data_folder (str): Folder containing NIfTI files.

    Returns:
        dict: Average surface areas and standard deviations per class.
    """
    # Load and filter the Excel file for 'good' quality scans
    excel_data = pd.read_excel(excel_path)
    good_quality_data = excel_data[excel_data['quality'] == 'good']

    # Initialize lists for surface areas
    healthy_surfaces = []
    pathological_surfaces = []

    # Iterate over rows to calculate surfaces
    for _, row in good_quality_data.iterrows():
        patient_id = row['Nr']
        classification = row['Classification']
        file_path = os.path.join(data_folder, f"{patient_id}.nii.gz")

        if os.path.exists(file_path):
            surface_area = calculate_heart_surface(file_path)
            if classification == 'healthy':
                healthy_surfaces.append(surface_area)
            elif classification == 'pathological':
                pathological_surfaces.append(surface_area)
        else:
            print(f"File not found: {file_path}")

    # Calculate averages and standard deviations
    avg_healthy_surface = np.mean(healthy_surfaces) if healthy_surfaces else 0
    std_healthy_surface = np.std(healthy_surfaces) if healthy_surfaces else 0
    avg_pathological_surface = np.mean(pathological_surfaces) if pathological_surfaces else 0
    std_pathological_surface = np.std(pathological_surfaces) if pathological_surfaces else 0

    # Print results
    print(f"Average Healthy Heart Surface: {avg_healthy_surface:.2f} mm² (± {std_healthy_surface:.2f} mm²)")
    print(f"Average Pathological Heart Surface: {avg_pathological_surface:.2f} mm² (± {std_pathological_surface:.2f} mm²)")

    # Improved Boxplot
    plt.figure(figsize=(8, 6))
    boxprops = dict(color='darkblue', linewidth=2)
    medianprops = dict(color='darkblue', linewidth=2)
    whiskerprops = dict(color='darkblue', linewidth=1.5)
    capprops = dict(color='darkblue', linewidth=2)

    plt.boxplot([healthy_surfaces, pathological_surfaces],
                labels=['Healthy', 'Pathological'],
                boxprops=boxprops, medianprops=medianprops,
                whiskerprops=whiskerprops, capprops=capprops, patch_artist=True,
                widths=0.6)

    plt.title('Heart Surface Comparison', fontsize=16)
    plt.ylabel('Surface Area (mm²)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig('heart_surface_boxplot.png', dpi=300)
    print("Boxplot saved as 'heart_surface_boxplot.png' in the current directory.")

    return {
        "Healthy": {"Mean": avg_healthy_surface, "Std": std_healthy_surface},
        "Pathological": {"Mean": avg_pathological_surface, "Std": std_pathological_surface}
    }

# Input paths
excel_path = "/home/fit_member/Documents/NS_SemesterWork/Project/data/data_overview_binary_cleaned_256.xlsx"
data_folder = "/home/fit_member/Documents/NS_SemesterWork/Project/data_final_without_aorta"

# Run the function
average_surfaces = calculate_average_surfaces(excel_path, data_folder)
