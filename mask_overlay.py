import os
import nibabel as nib
import numpy as np


def create_overlay(CT_directory, mask_directory, output_directory):
    """
    Create overlay of CT image with semi-transparent mask, darkening the background.
    
    Args:
        CT_directory (str): Path to directory containing CT images
        mask_directory (str): Path to directory containing binary masks in subdirectories
        output_directory (str): Path to directory where processed images will be saved
    """
    
    # Loop over all CT images
    for CT_filename in os.listdir(CT_directory):
        if CT_filename.endswith('.nii.gz'):
            CT_path = os.path.join(CT_directory, CT_filename)
            
            # Extract the base name (without extension) from CT file to find matching mask directory
            CT_base_name = os.path.splitext(os.path.splitext(CT_filename)[0])[0]
            
            # Search for a corresponding subdirectory in the mask directory
            for subdir in os.listdir(mask_directory):
                subdir_path = os.path.join(mask_directory, subdir)
                
                if os.path.isdir(subdir_path) and CT_base_name in subdir:
                    # Now look for the specific mask file (e.g., heart.nii.gz) inside the subdirectory
                    for mask_filename in os.listdir(subdir_path):
                        if CT_filename.endswith('.nii.gz'):
                            mask_path = os.path.join(subdir_path, mask_filename)
                            
                            # Load the CT image and the corresponding mask
                            CT_img = nib.load(CT_path)
                            mask_img = nib.load(mask_path)
                            
                            # Get the data from both images
                            CT_data = CT_img.get_fdata()
                            mask_data = mask_img.get_fdata()
                            
                            # Create an overlay where mask is semi-transparent (background is darkened)
                            overlay = np.where(mask_data == 1, CT_data, CT_data * 0.1)
                            
                            # Create a new NIfTI image for the overlay
                            overlay_image = nib.Nifti1Image(overlay, CT_img.affine)
                            
                            # Save the final overlay image in the output directory
                            output_path = os.path.join(output_directory, CT_filename)
                            nib.save(overlay_image, output_path)
                            
                            print(f'Processed and saved overlay: {output_path}')
                            break  # Exit after processing the first matching mask

                    break  # Break after processing the matching subdirectory

if __name__ == '__main__':
    
    CT_directory = 'Cropped_images/'
    mask_directory = 'Segmentation/'
    output_directory = 'CT_mask_overlay/'
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    create_overlay(CT_directory, mask_directory, output_directory)
