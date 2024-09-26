import nibabel as nib
import numpy as np
import os


def mask_overlay(CT_directory, mask_directory, output_directory):
    """
    Multiply the binary masks with the unprocessed CT images

    Args:
        CT_directory (str): Path to directory containing CT images
        mask_directory (str): Path to directory containing binary masks
        output_directory (str): Path to directory where processed images will be saved
    """
    
    for CT_filename in os.listdir(CT_directory):
        if CT_filename.endswith('.nii.gz'):
            CT_path = os.path.join(CT_directory, CT_filename)
            mask_path = os.path.join(mask_directory)
            
            CT_img = nib.load(CT_path)
            mask_img = nib.load(mask_path)
            
            CT_data = CT_img.get_fdata()
            mask_data = mask_img.get_fdata()
            
            
            # Replace 0 in the mask by NaN
            mask_data_nan = np.where(mask_data == 0, np.nan, mask_data)
            
            # Multiplication of CT image with mask
            final = CT_data * mask_data_nan
            
            # Create a final NIFTI image
            final_image = nib.Nifti1Image(final, CT_img.affine)
            
            # Save
            output_path = os.path.join(output_directory, CT_filename)
            nib.save(final_image, output_path)
            
            
            print(f'Processed and saved: {output_path}')
            
            
if __name__ == '__main__':
    
    CT_directory = 'Cropped_images/'
    mask_directory = 'Segmentation/THORAX_1_5_B31S_0002_IRM_Thorax_Abdomen_Routine_20120221083140_2.nii/heart.nii.gz/'
    output_directory = 'CT_with_mask/'
    
    mask_overlay(CT_directory, mask_directory, output_directory)

    
    
    
        
        
            