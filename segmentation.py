# This file has 1 function: segment_images().
# It takes a NIFTI file as input and saves a binary mask for the declared object of interest.

import nibabel as nib
from totalsegmentator import python_api
import torch
import os



def segment_images(input_directory, output_directory): # when using the 'total' task, add: object_of_interest parameter
    
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
            output_path = os.path.join(output_directory, os.path.splitext(filename)[0])
            
            # Run TotalSegmentator for each NIfTI file
            print(f'Processing file: {input_path}')
            python_api.totalsegmentator(input_path, output_path, task = 'heartchambers_highres') # without GPU use: fastest = True
                                                                                                                                                  # for the whole heart use: task = 'total'
                                                                                                                                                  # for the 'total' task, add roi_subset = object_of_interest as parameter
            

if __name__ == '__main__':
    #Check if GPU is available
    print(torch.cuda.is_available())
    
    input_directory = 'NIFTI_files/'
    output_directory = 'masks_heart_chambers/' # for the whole heart use: 'masks_heart/'
    #object_of_interest = ['heart'] only need if doing the 'total' task
    
    segment_images(input_directory, output_directory) # add: object_of_interest when doing the task 'total'
    
    
 
    
