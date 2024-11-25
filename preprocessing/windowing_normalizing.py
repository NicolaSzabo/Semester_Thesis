import os
import nibabel as nib
import numpy as np

# Define the windowing and normalization parameters
WINDOW_CENTER = 40
WINDOW_WIDTH = 400
LOWER_BOUND = WINDOW_CENTER - WINDOW_WIDTH / 2
UPPER_BOUND = WINDOW_CENTER + WINDOW_WIDTH / 2

# Define input and output directories
input_dir = 'G://semester_thesis//Project//data//data_classification//healthy_resized'
output_dir = 'G://semester_thesis//Project//data//final//healthy'

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)



# Function to apply windowing and normalization
def window_and_normalize(image, lower, upper):
    """
    Applies windowing and normalization to the image.
    Background (NaN) remains NaN.
    """
    # Preserve NaN in the background
    normalized_image = (image - lower) / (upper - lower)
    normalized_image = np.clip(normalized_image, 0, 1)

    # Ensure background (NaN) remains NaN
    normalized_image[np.isnan(image)] = np.nan

    return normalized_image




# Iterate through files in the input directory
for file_name in os.listdir(input_dir):
    # Process only NIfTI files
    if file_name.endswith(".nii.gz"):
        # Load the NIfTI file
        input_path = os.path.join(input_dir, file_name)
        nifti = nib.load(input_path)
        image_data = nifti.get_fdata()

        # Apply windowing and normalization
        processed_image_data = window_and_normalize(image_data, LOWER_BOUND, UPPER_BOUND)

        # Create the new filename
        base_name = file_name.split("_")[0]
        output_file_name = f"{base_name}_healthy.nii.gz"
        output_path = os.path.join(output_dir, output_file_name)

        # Save the processed image
        processed_nifti = nib.Nifti1Image(processed_image_data, affine=nifti.affine)
        nib.save(processed_nifti, output_path)

        print(f"Processed and saved: {file_name} -> {output_file_name}")