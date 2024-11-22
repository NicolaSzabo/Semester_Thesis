import os
import numpy as np
import nibabel as nib

# Function to compute summary statistics for a folder
def compute_statistics(folder_path):
    stats = {"mean": [], "median": [], "min": [], "max": []}

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".nii") or file_name.endswith(".nii.gz"):
            file_path = os.path.join(folder_path, file_name)
            
            # Load the NIfTI file
            img = nib.load(file_path)
            data = img.get_fdata()

            # Ignore NaN values
            valid_data = data[~np.isnan(data)]
            
            # Compute statistics
            stats["mean"].append(np.mean(valid_data))
            stats["median"].append(np.median(valid_data))
            stats["min"].append(np.min(valid_data))
            stats["max"].append(np.max(valid_data))

    # Aggregate statistics across all files
    aggregated_stats = {
        "mean": np.mean(stats["mean"]),
        "median": np.mean(stats["median"]),
        "min": np.min(stats["min"]),
        "max": np.max(stats["max"]),
    }
    return aggregated_stats

# Define folder paths
healthy_folder = 'G://semester_thesis//Project//data//final/healthy'
unhealthy_folder = 'G://semester_thesis//Project//data//final/unhealthy'

# Compute statistics for each folder
healthy_stats = compute_statistics(healthy_folder)
unhealthy_stats = compute_statistics(unhealthy_folder)

# Print the results
print("Healthy Folder Statistics:", healthy_stats)
print("Unhealthy Folder Statistics:", unhealthy_stats)


