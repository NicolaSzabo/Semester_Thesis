import os
import nibabel as nib
import numpy as np


def check_nans_in_directory(directory_path):
    """
    Check for NaN values in all .nii.gz files in a given directory.

    Args:
        directory_path (str): Path to the directory containing .nii.gz files.

    Returns:
        None
    """
    files_with_nans = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.nii.gz'):
            file_path = os.path.join(directory_path, filename)
            try:
                # Load the NIfTI file
                img = nib.load(file_path)
                data = img.get_fdata()

                # Check for NaNs in the data
                if np.isnan(data).any():
                    print(f"NaNs found in file: {file_path}")
                    files_with_nans.append(filename)
                else:
                    print(f"No NaNs in file: {file_path}")

            except Exception as e:
                print(f"Error loading file {file_path}: {e}")

    # Summary
    if files_with_nans:
        print("\nFiles with NaNs:")
        for file in files_with_nans:
            print(file)
    else:
        print("\nNo NaNs found in any file.")


# Directory path containing .nii.gz files
directory_path = '/home/fit_member/Documents/NS_SemesterWork/data/data_classification/unhealthy_final'
check_nans_in_directory(directory_path)