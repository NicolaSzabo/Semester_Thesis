# This file has 1 function: segment_images().
# It takes a NIFTI file as input and saves a binary mask for the declared object of interest.

import nibabel as nib
from totalsegmentator import python_api
import torch
import os



def segment_images(input_directory, output_directory, object_of_interest): 
    
    """
    Segmentate the cropped NIFTI files into objects of interest
    
    Parameters:
    input_directory (str): Path to directory containing the cropped NIFTI files
    output_directory (str): Path to directory to save segmentated NIFTI files
    object_of_interest (list): Specifying which object(s) should be segmentated
    """
    
    # Loop through all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.nii.gz'):
            input_path = os.path.join(input_directory, filename)

            # Extract the first 6 characters from the filename (xx-xxxx)
            base_filename = filename[:7]  # Extract the first 7 characters, which includes 'xx-xxxx'
            
            # Create a descriptive filename by appending the object of interest to the base filename
            custom_filename = f"{base_filename}_{'_'.join(object_of_interest)}.nii.gz"  # Add the objects of interest
            output_path = os.path.join(output_directory, custom_filename)
            
            # Run TotalSegmentator for each NIfTI file
            print(f'Processing file: {input_path} -> Output: {output_path}')
            python_api.totalsegmentator(input_path, output_path, task = 'total', roi_subset = object_of_interest)
                                                                                                                                                
                                                                                                                                                  
            

if __name__ == '__main__':
    
    #Check if GPU is available
    print(torch.cuda.is_available())
    
    input_directory = 'G:/data/unhealthy_nifti/'
    output_directory = 'G:/data/unhealthy_segmentation' 
    object_of_interest = ['heart', 'aorta'] 
    
    segment_images(input_directory, output_directory, object_of_interest) 
    
    
 
    