import nibabel as nib
import numpy as np
import os


def combine_masks(mask_subdirectory):
    """
    Combine heart and aorta masks into a single binary mask.

    Args:
        mask_subdirectory (str): Path to the directory containing heart and aorta masks.

    Returns:
        np.ndarray: Combined mask (heart and aorta combined into one binary mask).
    """
    heart_mask_path = os.path.join(mask_subdirectory, 'heart.nii.gz')
    aorta_mask_path = os.path.join(mask_subdirectory, 'aorta.nii.gz')

    # Load heart and aorta masks
    heart_mask = nib.load(heart_mask_path).get_fdata()
    aorta_mask = nib.load(aorta_mask_path).get_fdata()

    # Combine the masks using a logical OR (combines the regions of both heart and aorta)
    combined_mask = np.logical_or(heart_mask, aorta_mask).astype(np.uint8)

    return combined_mask


def mask_overlay(CT_directory, mask_directory, output_directory, label_suffix):
    """
    Multiply the combined binary masks (heart + aorta) with the unprocessed CT images.

    Args:
        CT_directory (str): Path to directory containing CT images.
        mask_directory (str): Path to directory containing binary masks in subdirectories.
        output_directory (str): Path to directory where processed images will be saved.
    """

    # Get list of all NIfTI files (both CT and mask files)
    CT_files = [f for f in os.listdir(CT_directory) if f.endswith('.nii.gz')]

    for CT_filename in CT_files:
        CT_path = os.path.join(CT_directory, CT_filename)
        patient_id = CT_filename[:7]  # Extract patient ID (first 7 characters) e.g., '11-1382'

        # Locate the corresponding mask subdirectory for the patient
        mask_subdirectory = os.path.join(mask_directory, f"{patient_id}_heart_aorta.nii.gz")

        if os.path.isdir(mask_subdirectory):
            # Combine the heart and aorta masks
            combined_mask_data = combine_masks(mask_subdirectory)

            # Load the CT image
            CT_img = nib.load(CT_path)
            CT_data = CT_img.get_fdata()

            # Replace 0 in the combined mask with NaN
            combined_mask_nan = np.where(combined_mask_data == 0, np.nan, combined_mask_data)

            # Multiply the CT image by the combined mask (keeping only the regions of interest)
            final = CT_data * combined_mask_nan

            # Create a new NIfTI image for the final masked CT
            final_image = nib.Nifti1Image(final, CT_img.affine)


            # Create a custom filename with patient ID and the provided label suffix
            custom_filename = f"{patient_id}_{label_suffix}.nii.gz"
            output_path = os.path.join(output_directory, custom_filename)

            # Save the final masked CT image
            nib.save(final_image, output_path)

            print(f'Processed and saved: {output_path}')
        else:
            print(f"Mask directory for {patient_id} not found.")


if __name__ == '__main__':

    CT_directory = '/home/fit_member/Documents/NS_SemesterWork/data/unhealthy_nifti'  # Directory with CT images
    mask_directory = '/home/fit_member/Documents/NS_SemesterWork/data/unhealthy_segmentation'  # Directory with heart and aorta masks in subfolders
    output_directory = '/home/fit_member/Documents/NS_SemesterWork/data/unhealthy_final'  # Output directory for masked CT images

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    mask_overlay(CT_directory, mask_directory, output_directory, label_suffix = 'unhealthy')
    

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    mask_overlay(CT_directory, mask_directory, output_directory)

        
        
            