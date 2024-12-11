import nibabel as nib
from totalsegmentator import python_api
import torch
import os


def segment_images(input_directory, output_directory): 
    """
    Segment the cropped NIFTI files into objects of interest, skipping already processed files.
    
    Parameters:
    input_directory (str): Path to directory containing the cropped NIFTI files
    output_directory (str): Path to directory to save segmented NIFTI files
    """
    
    # Loop through all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.nii.gz'):
            input_path = os.path.join(input_directory, filename)

            # Extract the first 7 characters from the filename (xx-xxxx)
            base_filename = filename[:7]  # Extract the first 7 characters
            custom_filename = f"{base_filename}.nii.gz"  # Create the output filename
            output_path = os.path.join(output_directory, custom_filename)
            
            # Check if the output file already exists
            if os.path.exists(output_path):
                print(f"Skipping file {input_path} as {output_path} already exists.")
                continue
            
            # Process the file if not already processed
            print(f'Processing file: {input_path} -> Output: {output_path}')
            python_api.totalsegmentator(input_path, output_path, task='heartchambers_highres')
            

if __name__ == '__main__':
    # Check if GPU is available
    print(torch.cuda.is_available())
    
    input_directory = "G://data//niftis_heart"
    # '/home/fit_member/Documents/NS_SemesterWork/Project/data/niftis_heart'
    output_directory = "G://data//segmentation_heart_specific"
    # '/home/fit_member/Documents/NS_SemesterWork/Project/data/segmentation_heart'
    
    # Call the segmentation function
    segment_images(input_directory, output_directory)
