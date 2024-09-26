import nibabel as nib
from totalsegmentator import python_api
import torch
import os



def segment_images(input_directory, output_directory, object_of_interst):
    
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
            python_api.totalsegmentator(input_path, output_path, roi_subset = object_of_interest) # without GPU use: fastest = True


if __name__ == '__main__':
    #Check if GPU is available
    print(torch.cuda.is_available())
    
    input_directory = 'Cropped_images/'
    output_directory = 'Segmentation/'
    object_of_interest = ['heart']
    
    segment_images(input_directory, output_directory, object_of_interest)
    
    
 
    
