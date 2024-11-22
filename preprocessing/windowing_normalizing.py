import os
import nibabel as nib
import numpy as np

# Define the windowing and normalization parameters
WINDOW_CENTER = 40
WINDOW_WIDTH = 400
LOWER_BOUND = WINDOW_CENTER - WINDOW_WIDTH / 2
UPPER_BOUND = WINDOW_CENTER + WINDOW_WIDTH / 2

# Define the root directory and subfolders
root_dir = 'G://semester_thesis//Project//data//final'
subfolders = ['healthy', 'unhealthy']

# Function to apply windowing and normalization
def window_and_normalize(image, mask, lower, upper):
    """
    Applies windowing and normalization to the image within the mask region.
    Background remains 0.
    """
    # Apply mask to retain only the heart region
    masked_image = np.where(mask > 0, image, np.nan)
    
    # Normalize the masked region to [0, 1]
    normalized_image = np.where(mask > 0, (masked_image - lower) / (upper - lower), np.nan)
    
    normalized_image = np.clip(normalized_image, 0, 1)
    
    return normalized_image

# Iterate through subfolders
for subfolder in subfolders:
    folder_path = os.path.join(root_dir, subfolder)
    for file_name in os.listdir(folder_path):
        # Process only NIfTI files
        if file_name.endswith(".nii_processed.nii.gz"):
            # Load the NIfTI file
            file_path = os.path.join(folder_path, file_name)
            nifti = nib.load(file_path)
            image_data = nifti.get_fdata()

            # Use the same image as the mask (only non-zero regions are the heart)
            mask = np.where(image_data > 0, 1, 0)

            # Apply windowing and normalization
            processed_image_data = window_and_normalize(image_data, mask, LOWER_BOUND, UPPER_BOUND)

            # Create the new filename
            base_name = file_name.split("_")[0]  # Extract xx-xxxx
            new_file_name = f"{base_name}_{subfolder}.nii.gz"
            new_file_path = os.path.join(folder_path, new_file_name)

            # Save the processed image
            processed_nifti = nib.Nifti1Image(processed_image_data, affine=nifti.affine)
            nib.save(processed_nifti, new_file_path)

            # Remove the old file
            os.remove(file_path)

            print(f"Processed and replaced: {file_name} -> {new_file_name}")
