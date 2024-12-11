import numpy as np
import nibabel as nib
import os

# Define the list of relevant masks for cropping
RELEVANT_MASKS = [
    "heart_atrium_left.nii.gz",
    "heart_atrium_right.nii.gz",
    "heart_myocardium.nii.gz",
    "heart_ventricle_left.nii.gz",
    "heart_ventricle_right.nii.gz",
    "pulmonary_artery.nii.gz"
]

EXCLUDE_FROM_CROP = ["aorta.nii.gz"]  # Masks to include in values but exclude from cropping

def crop_around_mask(data, combined_mask, padding=50):
    """
    Crop the data around the non-zero values of the combined mask.
    If no valid non-zero values are found, return None to signal skipping.
    """
    # Find the non-zero coordinates in the combined mask
    non_zero_coords = np.where(combined_mask > 0)

    # Handle case where no non-zero values are found
    if len(non_zero_coords[0]) == 0:
        print("Warning: No non-zero values in cropping mask. Skipping this file.")
        return None, None, None  # Return None to signal skipping

    # Calculate the bounding box with padding
    x_min = max(non_zero_coords[0].min() - padding, 0)
    x_max = min(non_zero_coords[0].max() + padding, data.shape[0] - 1)
    y_min = max(non_zero_coords[1].min() - padding, 0)
    y_max = min(non_zero_coords[1].max() + padding, data.shape[1] - 1)
    z_min = max(non_zero_coords[2].min() - padding, 0)
    z_max = min(non_zero_coords[2].max() + padding, data.shape[2] - 1)

    # Crop the data using the bounding box
    cropped_data = data[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
    cropped_mask = combined_mask[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]

    return cropped_data, cropped_mask, (x_min, x_max, y_min, y_max, z_min, z_max)




def mask_overlay_with_dynamic_crop(CT_directory, mask_directory, output_directory, padding=50):
    # List all CT files
    CT_files = [f for f in os.listdir(CT_directory) if f.endswith('.nii.gz')]

    for CT_filename in CT_files:
        CT_path = os.path.join(CT_directory, CT_filename)
        patient_id = CT_filename.replace('.nii.gz', '')  # Get patient ID from the filename

        # Construct the path to the patient’s mask folder (appending .nii.gz)
        patient_mask_folder = os.path.join(mask_directory, f"{patient_id}.nii.gz")

        # Combine relevant masks
        combined_mask = None
        cropping_mask = None
        if os.path.isdir(patient_mask_folder):
            mask_files = os.listdir(patient_mask_folder)
            for mask_file in mask_files:
                if mask_file in EXCLUDE_FROM_CROP:  # Skip excluded masks
                    continue
                
                mask_path = os.path.join(patient_mask_folder, mask_file)
                mask_data = nib.load(mask_path).get_fdata()

                # Include in the combined mask
                if combined_mask is None:
                    combined_mask = mask_data
                else:
                    combined_mask = np.maximum(combined_mask, mask_data)

                # Include only RELEVANT_MASKS in cropping
                if mask_file in RELEVANT_MASKS:
                    if cropping_mask is None:
                        cropping_mask = mask_data
                    else:
                        cropping_mask = np.maximum(cropping_mask, mask_data)

        # Ensure we have a cropping mask
        if cropping_mask is None:
            print(f"No valid cropping masks found for {patient_id}. Skipping.")
            continue

        # Load the CT image
        CT_img = nib.load(CT_path)
        CT_data = CT_img.get_fdata()

        # Apply the combined mask to retain only the values inside the mask
        masked_CT = np.where(combined_mask > 0, CT_data, np.nan)

         # Crop the masked CT image and mask based on the cropping mask
        cropped_CT, cropped_mask, bbox = crop_around_mask(masked_CT, cropping_mask, padding=padding)

        # Skip the file if no valid cropping mask was found
        if cropped_CT is None or cropped_mask is None:
            print(f"Skipping {CT_filename} due to no valid cropping mask.")
            continue

        # Save the cropped CT image
        cropped_img = nib.Nifti1Image(cropped_CT, CT_img.affine)
        output_path = os.path.join(output_directory, f"{patient_id}.nii.gz")
        nib.save(cropped_img, output_path)
        print(f"Processed and saved: {output_path} with bounding box {bbox}")

if __name__ == '__main__':
    CT_directory = "G://data//niftis_full_body"
    mask_directory = "G://data//segmentation_heart"
    output_directory = "G://data//preprocessing//masked_without_aorta"
    padding = 0  # Set padding around the ROI

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Run the function
    mask_overlay_with_dynamic_crop(
        CT_directory=CT_directory,
        mask_directory=mask_directory,
        output_directory=output_directory,
        padding=padding
    )
