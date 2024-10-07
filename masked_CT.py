# This file has 1 function: mask_overlay().
# It loads the .nii.gz files of the masks and the original CT images from 2 different directories and multiplies them with each other.
# So in the output is a non-binary mask which has the original CT values stored.

import nibabel as nib
import numpy as np
import os


def mask_overlay(CT_directory, mask_directory, output_directory):
    """
    Multiply the binary masks with the unprocessed CT images
    
    Args:
        CT_directory (str): Path to directory containing CT images
        mask_directory (str): Path to directory containing binary masks in subdirectories
        output_directory (str): Path to directory where processed images will be saved
    """
    
    
    mask_files = [f for f in os.listdir(mask_directory) if f.endswith('.nii.gz')]
    CT_files = [f for f in os.listdir(CT_directory) if f.endswith('.nii.gz')]
                
                    
    for mask_filename, CT_filename in zip(mask_files, CT_files):
        
        
        mask_path = os.path.join(mask_directory, mask_filename)            
        CT_path = os.path.join(CT_directory, CT_filename)
                 
        # Load the CT image and the corresponding mask
        CT_img = nib.load(CT_path)
        mask_img = nib.load(mask_path)
                            
        # Get the data from both images
        CT_data = CT_img.get_fdata()
        mask_data = mask_img.get_fdata()
                            
        # Replace 0 in the mask by NaN
        mask_data_nan = np.where(mask_data == 0, np.nan, mask_data)
                            
        # Multiply the CT image by the mask
        final = CT_data * mask_data_nan
                            
        # Create a new NIfTI image with the result
        final_image = nib.Nifti1Image(final, CT_img.affine)
                            
        # Save the final image in the output directory
        output_path = os.path.join(output_directory, CT_filename)
        nib.save(final_image, output_path)
                            

        print(f'Processed and saved: {output_path}')
                            
                            
 

if __name__ == '__main__':
    
    CT_directory = 'NIFTI_files/' 
    mask_directory = 'masks_heart/'
    output_directory = 'masked_CT/'
    
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    mask_overlay(CT_directory, mask_directory, output_directory)

        
        
            