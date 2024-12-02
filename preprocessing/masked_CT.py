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

def crop_around_mask(data, mask, reference_masks=None, padding=50):
    if reference_masks:
        # Combine all reference masks to determine the bounding box
        combined_reference_mask = np.zeros(mask.shape)
        for ref_mask in reference_masks:
            combined_reference_mask = np.maximum(combined_reference_mask, ref_mask)
        non_zero_coords = np.where(combined_reference_mask > 0)
    else:
        # Use the main mask if no reference masks are provided
        non_zero_coords = np.where(mask > 0)

    # Calculate the bounding box with padding
    x_min = max(non_zero_coords[0].min() - padding, 0)
    x_max = min(non_zero_coords[0].max() + padding, mask.shape[0] - 1)
    y_min = max(non_zero_coords[1].min() - padding, 0)
    y_max = min(non_zero_coords[1].max() + padding, mask.shape[1] - 1)
    z_min = max(non_zero_coords[2].min() - padding, 0)
    z_max = min(non_zero_coords[2].max() + padding, mask.shape[2] - 1)

    # Crop the data using the bounding box
    cropped_data = data[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]

    return cropped_data, (x_min, x_max, y_min, y_max, z_min, z_max)

def mask_overlay_with_dynamic_crop(CT_directory, mask_directory, output_directory, padding=5):
    # List all CT files
    CT_files = [f for f in os.listdir(CT_directory) if f.endswith('.nii.gz')]

    for CT_filename in CT_files:
        CT_path = os.path.join(CT_directory, CT_filename)
        patient_id = CT_filename.replace('.nii.gz', '')  # Get patient ID from the filename

        # Construct the path to the patient’s mask folder (appending .nii.gz)
        patient_mask_folder = os.path.join(mask_directory, f"{patient_id}.nii.gz")

        # Load and combine relevant masks for cropping
        reference_masks = []
        if os.path.isdir(patient_mask_folder):
            mask_files = os.listdir(patient_mask_folder)
            for mask_file in mask_files:
                if mask_file in RELEVANT_MASKS:  # Include only relevant masks
                    mask_path = os.path.join(patient_mask_folder, mask_file)
                    mask_data = nib.load(mask_path).get_fdata()
                    reference_masks.append(mask_data)

        # Ensure we have at least one reference mask
        if not reference_masks:
            print(f"No valid reference masks found for {patient_id}. Using full CT volume.")
            reference_masks = [np.zeros_like(nib.load(CT_path).get_fdata())]

        # Load the CT image
        CT_img = nib.load(CT_path)
        CT_data = CT_img.get_fdata()

        # Crop the original CT image using the reference masks
        cropped_CT, bbox = crop_around_mask(CT_data, None, reference_masks=reference_masks, padding=padding)

        # Save the cropped CT image
        cropped_img = nib.Nifti1Image(cropped_CT, CT_img.affine)
        output_path = os.path.join(output_directory, f"{patient_id}.nii.gz")
        nib.save(cropped_img, output_path)
        print(f"Processed and saved: {output_path} with bounding box {bbox}")

if __name__ == '__main__':
    CT_directory = 'Project/data/niftis_full_body'
    mask_directory = 'Project/data/segmentation_heart'
    output_directory = 'Project/data/masked_with_nan'
    padding = 10  # Set padding around the ROI

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
