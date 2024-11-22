import numpy as np
import nibabel as nib
import os

def crop_around_mask(data, mask):
    """
    Crop the 3D image data around the region where the mask is non-zero.
    Args:
        data (np.ndarray): The input 3D array (CT image).
        mask (np.ndarray): The binary mask indicating the region of interest.
    Returns:
        tuple: The cropped data and the coordinates of the bounding box.
    """
    # Find the non-zero indices in the mask
    non_zero_coords = np.where(~np.isnan(mask))  # Use ~np.isnan to locate valid regions

    # Check if the mask is empty (no non-NaN values)
    if len(non_zero_coords[0]) == 0:
        print("Warning: Mask is empty; skipping this image.")
        return None, None, None

    # Calculate the bounding box
    x_min, x_max = non_zero_coords[0].min(), non_zero_coords[0].max()
    y_min, y_max = non_zero_coords[1].min(), non_zero_coords[1].max()
    z_min, z_max = non_zero_coords[2].min(), non_zero_coords[2].max()

    # Crop the data using the bounding box
    cropped_data = data[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
    cropped_mask = mask[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]

    return cropped_data, cropped_mask, (x_min, x_max, y_min, y_max, z_min, z_max)







def mask_overlay_with_dynamic_crop(CT_directory, mask_directory, output_directory, label_suffix):
    """
    Overlay the mask on the CT image, crop around the mask, and save the result.
    Args:
        CT_directory (str): Path to directory containing CT images.
        mask_directory (str): Path to directory containing binary masks.
        output_directory (str): Path to directory where processed images will be saved.
        label_suffix (str): A suffix to append to the output filename ('_healthy' or '_unhealthy').
    """
    CT_files = [f for f in os.listdir(CT_directory) if f.endswith('.nii.gz')]

    for CT_filename in CT_files:
        CT_path = os.path.join(CT_directory, CT_filename)
        patient_id = CT_filename[:7]  # Assuming patient ID is the first 7 characters

        # Construct the path to the patient’s mask folder
        patient_mask_folder = os.path.join(mask_directory, f"{patient_id}_heart.nii.gz")

        # Check if the folder exists
        if os.path.isdir(patient_mask_folder):
            
            # Define paths for the heart mask inside the folder
            heart_mask_path = os.path.join(patient_mask_folder, 'heart.nii.gz')

            # Check that the mask file exists
            if os.path.isfile(heart_mask_path):
                # Load the mask
                heart_mask = nib.load(heart_mask_path).get_fdata()

                # Convert the mask to 1 and NaN
                nan_mask = np.where(heart_mask > 0, 1, np.nan)
                
                # Load the CT image
                CT_img = nib.load(CT_path)
                CT_data = CT_img.get_fdata()

                # Apply the mask to the CT image
                masked_CT = CT_data * nan_mask


                # Crop the CT image around the mask
                cropped_CT, cropped_mask, bbox = crop_around_mask(masked_CT, nan_mask)

                # Check if cropping was successful
                if cropped_CT is None:
                    print(f"Skipping {CT_filename} due to empty mask.")
                    continue

                # Proceed with saving if the crop was successful
                cropped_img = nib.Nifti1Image(cropped_CT, CT_img.affine)
                output_path = os.path.join(output_directory, f"{patient_id}_{label_suffix}.nii.gz")
                nib.save(cropped_img, output_path)
                print(f"Processed and saved: {output_path} with bounding box {bbox}")

            else:
                print(f"Mask file not found in {patient_mask_folder} for {patient_id}.")
        else:
            print(f"Folder for {patient_id} not found at {patient_mask_folder}")




if __name__ == '__main__':
    # Define the directories for your CT images, masks, and output location
    CT_directory = 'G://semester_thesis//Project//data//healthy_nifti'
    mask_directory = 'G://semester_thesis//Project//data//healthy_segmentation'
    output_directory = 'G://semester_thesis//Project//data//data_classification//healthy_final'
    label_suffix = 'healthy'  # Suffix to differentiate healthy vs. unhealthy images

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Call the mask overlay and dynamic cropping function
    mask_overlay_with_dynamic_crop(
        CT_directory=CT_directory,
        mask_directory=mask_directory,
        output_directory=output_directory,
        label_suffix=label_suffix
    )
