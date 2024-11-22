import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# Path to the data directory
data_path = 'G://semester_thesis//Project//data//final'

# Initialize dictionaries to store results for each class
results = {
    "healthy": {"min_values": [], "max_values": [], "nan_count": 0},
    "unhealthy": {"min_values": [], "max_values": [], "nan_count": 0}
}

# Loop through each class folder
for class_label in ['healthy', 'unhealthy']:
    class_path = os.path.join(data_path, class_label)
    
    # Loop through each NIfTI file in the folder
    for file_name in os.listdir(class_path):
        if file_name.endswith(".nii") or file_name.endswith(".nii.gz"):
            file_path = os.path.join(class_path, file_name)
            
            # Load the NIfTI file
            img = nib.load(file_path)
            data = img.get_fdata()
            
            # Check for NaN values
            nan_count = np.isnan(data).sum()
            if nan_count > 0:
                print(f"NaN values found in {file_name}: {nan_count}")
                results[class_label]["nan_count"] += nan_count
            
            # Calculate min and max
            min_value = np.nanmin(data)
            max_value = np.nanmax(data)
            results[class_label]["min_values"].append(min_value)
            results[class_label]["max_values"].append(max_value)

            # Plot histogram for the current file (optional)
            plt.hist(data[~np.isnan(data)].ravel(), bins=50, alpha=0.5, label=file_name)

# Summary statistics and histograms for each class
for class_label in ['healthy', 'unhealthy']:
    min_values = results[class_label]["min_values"]
    max_values = results[class_label]["max_values"]
    nan_count = results[class_label]["nan_count"]

    print(f"Class: {class_label}")
    print(f"Min values: {min_values}")
    print(f"Max values: {max_values}")
    print(f"Total NaN count: {nan_count}")
    print(f"Overall Min: {min(min_values)}")
    print(f"Overall Max: {max(max_values)}")

    # Combined histogram for the class
    plt.figure()
    plt.hist(min_values, bins=50, alpha=0.5, label="Min values")
    plt.hist(max_values, bins=50, alpha=0.5, label="Max values")
    plt.title(f"Histogram for {class_label}")
    plt.legend()
    plt.show()
