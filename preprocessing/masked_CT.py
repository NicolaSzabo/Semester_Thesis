import numpy as np
import nibabel as nib
import os

def crop_around_mask(data, mask, reference_masks=None):
 
    if reference_masks:
        # Combine all reference masks to determine the bounding box
        combined_reference_mask = np.zeros(mask.shape)
        for ref_mask in reference_masks:
            combined_reference_mask = np.maximum(combined_reference_mask, ref_mask)
        non_zero_coords = np.where(combined_reference_mask > 0)
    else:
        # Use the main mask if no reference masks are provided
        non_zero_coords = np.where(mask > 0)

    # Check if the mask is empty (no non-zero values)
    if len(non_zero_coords[0]) == 0:
        print("Warning: Reference mask is empty; skipping this image.")
        return None, None, None

    # Calculate the bounding box
    x_min, x_max = non_zero_coords[0].min(), non_zero_coords[0].max()
    y_min, y_max = non_zero_coords[1].min(), non_zero_coords[1].max()
    z_min, z_max = non_zero_coords[2].min(), non_zero_coords[2].max()

    # Crop the data using the bounding box
    cropped_data = data[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
    cropped_mask = mask[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]

    return cropped_data, cropped_mask, (x_min, x_max, y_min, y_max, z_min, z_max)









def mask_overlay_with_dynamic_crop(CT_directory, mask_directory, output_directory, label_suffix, reference_mask_names):
   
    CT_files = [f for f in os.listdir(CT_directory) if f.endswith('.nii.gz')]

    for CT_filename in CT_files:
        CT_path = os.path.join(CT_directory, CT_filename)
        patient_id = CT_filename[:7]  # Assuming patient ID is the first 7 characters

        # Construct the path to the patient’s mask folder
        patient_mask_folder = os.path.join(mask_directory, f"{patient_id}_masks")

        # Check if the folder exists
        if os.path.isdir(patient_mask_folder):
            mask_files = os.listdir(patient_mask_folder)

            # Load all reference masks
            reference_masks = []
            for mask_name in reference_mask_names:
                reference_mask_path = os.path.join(patient_mask_folder, f"{mask_name}.nii.gz")
                if not os.path.isfile(reference_mask_path):
                    print(f"Reference mask '{mask_name}' not found for {patient_id}. Skipping this mask.")
                    continue
                reference_masks.append(nib.load(reference_mask_path).get_fdata())

            # Ensure we have at least one valid reference mask
            if not reference_masks:
                print(f"No valid reference masks found for {patient_id}. Skipping.")
                continue

            # Load all masks and overlay them
            combined_mask = np.zeros(reference_masks[0].shape)
            for mask_file in mask_files:
                mask_path = os.path.join(patient_mask_folder, mask_file)
                mask_data = nib.load(mask_path).get_fdata()
                combined_mask = np.maximum(combined_mask, mask_data)

            # Convert the combined mask to binary
            binary_mask = np.where(combined_mask > 0, 1, np.nan)

            # Load the CT image
            CT_img = nib.load(CT_path)
            CT_data = CT_img.get_fdata()

            # Apply the mask to the CT image
            masked_CT = np.where(binary_mask == 1, CT_data, np.nan)  # Set background to nan

            # Crop the CT image and mask using the reference masks
            cropped_CT, cropped_mask, bbox = crop_around_mask(masked_CT, binary_mask, reference_masks=reference_masks)

            # Check if cropping was successful
            if cropped_CT is None:
                print(f"Skipping {CT_filename} due to empty reference masks.")
                continue

            # Save the cropped CT image
            cropped_img = nib.Nifti1Image(cropped_CT, CT_img.affine)
            output_path = os.path.join(output_directory, f"{patient_id}_{label_suffix}.nii.gz")
            nib.save(cropped_img, output_path)
            print(f"Processed and saved: {output_path} with bounding box {bbox}")

        else:
            print(f"Folder for {patient_id} not found at {patient_mask_folder}")






if __name__ == '__main__':
    CT_directory = 'G://semester_thesis//Project//data//nifti_files//unhealthy'
    mask_directory = 'G://data//segmentation//unhealthy'
    output_directory = 'G://data//preprocessing//unhealthy_masked_nan//unhealthy'
    label_suffix = 'unhealthy'
    reference_mask_names = ['heart', 'left_ventricle', 'right_ventricle']  # Specify multiple reference masks

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Run the function
    mask_overlay_with_dynamic_crop(
        CT_directory=CT_directory,
        mask_directory=mask_directory,
        output_directory=output_directory,
        label_suffix=label_suffix,
        reference_mask_names=reference_mask_names
    )
