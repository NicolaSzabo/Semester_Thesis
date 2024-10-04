# This file has 2 functions: get_image_dimensions(), crop_nifti_files().
# The first function is used to get the dimensions of a NIFTI file located in the input directory.
# The second function crops all NIFTI images located in a directory to a specified ROI. The cropped
# images are safe in a declared output directory.


import os
import nibabel as nib

def get_image_dimensions(input_directory):
    """
    Get the dimensions of the first NIfTI file in a directory.
    
    Parameters:
    input_directory (str): Path to the directory containing NIfTI files.
    
    Returns:
    tuple: A tuple containing the dimensions (x, y, z) of the first NIfTI file.
    """
    for filename in os.listdir(input_directory):
        if filename.endswith('.nii.gz'):
            input_path = os.path.join(input_directory, filename)
            
            # Load the NIfTI image
            img = nib.load(input_path)
            img_x, img_y, img_z = img.shape
            
            return img_x, img_y, img_z
        

def crop_nifti_files(input_directory, output_directory, roi):
    """
    Crop all NIfTI files in a directory to a specified ROI.
    
    Parameters:
    input_directory (str): Path to the directory containing NIfTI files.
    output_directory (str): Path to the directory to save cropped NIfTI files.
    roi (tuple): A tuple specifying the cropping bounds (x_start, x_end, y_start, y_end, z_start, z_end).
    """
    x_start, x_end, y_start, y_end, z_start, z_end = roi
    
    # Loop through all NIfTI files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.nii.gz'):
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, filename)
            
            # Load the NIfTI image
            img = nib.load(input_path)
            data = img.get_fdata()
            
            # Apply the same ROI cropping to each image
            cropped_data = data[x_start:x_end, y_start:y_end, z_start:z_end]

            # Create a new NIfTI image for the cropped data
            cropped_img = nib.Nifti1Image(cropped_data, img.affine)

            # Save the cropped image
            nib.save(cropped_img, output_path)

            print(f'Cropped and saved: {output_path}')
            

if __name__ == '__main__':
    # Define input and output directories
    
    #input_directory = 'NIFTI_files'
    #output_directory = 'cropped_images'
    input_directory = 'masks_andrea_uncropped/'
    output_directory = 'ground_truth_masks/'
    
    # Get image dimensions from the first image in the directory
    img_x, img_y, img_z = get_image_dimensions(input_directory)
    
    # Define the ROI based on the image dimensions
    #roi = (50, 400, 125, 375, 0, 250)  # Adjust the values based on your needs
    roi = (50, 400, 75, 325, 0, 250)
    
    # Call the function to crop all NIfTI files
    crop_nifti_files(input_directory, output_directory, roi)




