import os
import nibabel as nib
import numpy as np

def replace_nan_in_nifti(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate over all NIfTI files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".nii.gz"):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)
            
            # Load the NIfTI file
            img = nib.load(input_path)
            img_data = img.get_fdata()
            
            # Replace NaN values with 0
            img_data = np.nan_to_num(img_data, nan=0)
            
            # Save the corrected NIfTI file
            corrected_img = nib.Nifti1Image(img_data, img.affine, img.header)
            nib.save(corrected_img, output_path)
            print(f"Processed: {file_name}")

# Paths for healthy subfolder
input_path_healthy = '/home/fit_member/Documents/NS_SemesterWork/Project/data/preprocessing/windowed_normalized'
output_path_healthy = '/home/fit_member/Documents/NS_SemesterWork/Project/data_final'

# Process the healthy folder
replace_nan_in_nifti(input_path_healthy, output_path_healthy)
