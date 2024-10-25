import nibabel as nib
import numpy as np
import os

def adjust_dimensions(data, target_depth):
    """
    Adjust the z-axis of the data to a target depth.
    Crops if current depth is greater than target depth, or pads if less.

    Args:
        data (np.ndarray): The input 3D array (CT or mask) to be adjusted.
        target_depth (int): The desired depth along the z-axis.

    Returns:
        np.ndarray: The adjusted data with the specified z-axis depth.
    """
    current_depth = data.shape[2]
    
    # Crop the z-axis if it exceeds target depth
    if current_depth > target_depth:
        start = (current_depth - target_depth) // 2
        adjusted_data = data[:, :, start:start + target_depth]
        
    # Pad the z-axis if it's less than target depth
    elif current_depth < target_depth:
        pad_before = (target_depth - current_depth) // 2
        pad_after = target_depth - current_depth - pad_before
        adjusted_data = np.pad(data, ((0, 0), (0, 0), (pad_before, pad_after)), mode='constant', constant_values=0)
    else:
        adjusted_data = data  # No adjustment needed if depth matches
    
    return adjusted_data


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



def mask_overlay(CT_directory, mask_directory, output_directory, label_suffix, target_depth):
    """
    Multiply the combined binary masks (heart + aorta) with the unprocessed CT images.

    Args:
        CT_directory (str): Path to directory containing CT images.
        mask_directory (str): Path to directory containing binary masks in subdirectories.
        output_directory (str): Path to directory where processed images will be saved.
        label_suffix (str): A suffix to append to the output filename ('_healthy' or '_unhealthy').
        target_depth (int): Desired depth for the z-axis.
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

            # Adjust both the CT and the combined mask along the z-axis to match target depth
            final_adjusted = adjust_dimensions(final, target_depth)

            # Create a new NIfTI image for the final masked CT
            final_image = nib.Nifti1Image(final_adjusted, CT_img.affine)

            # Create a custom filename with patient ID and the provided label suffix
            custom_filename = f"{patient_id}_{label_suffix}.nii.gz"
            output_path = os.path.join(output_directory, custom_filename)

            # Save the final masked CT image
            nib.save(final_image, output_path)

            print(f'Processed and saved: {output_path}')
        else:
            print(f"Mask directory for {patient_id} not found.")



if __name__ == '__main__':

    #CT_directory = '/home/fit_member/Documents/NS_SemesterWork/data/unhealthy_nifti'  # Directory with CT images
    CT_directory = '/Users/nicolaszabo/Library/CloudStorage/OneDrive-Persönlich/Desktop/Semester_Thesis/Project/unhealthy_nifti'

    #mask_directory = '/home/fit_member/Documents/NS_SemesterWork/data/unhealthy_segmentation'  # Directory with heart and aorta masks in subfolders
    mask_directory = '/Users/nicolaszabo/Library/CloudStorage/OneDrive-Persönlich/Desktop/Semester_Thesis/Project/unhealthy_segmentation'

    #output_directory = '/home/fit_member/Documents/NS_SemesterWork/data/unhealthy_final'  # Output directory for masked CT images
    output_directory = '/Users/nicolaszabo/Library/CloudStorage/OneDrive-Persönlich/Desktop/Semester_Thesis/Project/unhealthy_final'


    target_depth = 240  # Set your target depth along the z-axis

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    mask_overlay(CT_directory, mask_directory, output_directory, label_suffix = 'unhealthy', target_depth = target_depth)
