import os
import nibabel as nib
import numpy as np

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
            output_path = os.path.join(output_directory, os.path.splitext(filename)[0])
            
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
    input_directory = 'NIFTI_files/'
    output_directory = 'Cropped_images/'
    
    # Define the ROI to crop, find the values in 3D slicer
    roi = (50, 450, 150, 450, 0, 200) # Possible values for heart, might be changed!  
    
    # Call the function to crop all NIfTI files
    crop_nifti_files(input_directory, output_directory, roi)





