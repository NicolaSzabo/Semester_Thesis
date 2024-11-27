import os
import nibabel as nib
import numpy as np


def find_max_dimensions(directory):
    """
    Find the maximum x, y, and z dimensions across all NIfTI images in a directory.
    Args:
        directory (str): Directory containing NIfTI images.
    Returns:
        tuple: Maximum x, y, and z dimensions.
    """
    max_x, max_y, max_z = 0, 0, 0

    # Iterate over each NIfTI file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.nii.gz'):
            file_path = os.path.join(directory, filename)

            # Load the image and get its data shape
            img = nib.load(file_path)
            data = img.get_fdata()
            x_dim, y_dim, z_dim = data.shape[0], data.shape[1], data.shape[2]

            # Update max_x, max_y, and max_z if the current image has larger dimensions
            max_x = max(max_x, x_dim)
            max_y = max(max_y, y_dim)
            max_z = max(max_z, z_dim)

            print(f"{filename} has dimensions (x, y, z): ({x_dim}, {y_dim}, {z_dim})")

    print(f"\nMaximum x dimension across all images: {max_x}")
    print(f"Maximum y dimension across all images: {max_y}")
    print(f"Maximum z dimension across all images: {max_z}")

    return max_x, max_y, max_z


def pad_to_target(data, target_shape):
    """
    Pad the x, y, and z dimensions of a 3D numpy array to make it match target_shape.
    Args:
        data (np.ndarray): The input 3D array (CT image).
        target_shape (tuple): The target size as (x, y, z).
    Returns:
        np.ndarray: The padded 3D array with dimensions matching target_shape.
    """
    x_dim, y_dim, z_dim = data.shape
    target_x, target_y, target_z = target_shape

    # Padding for x dimension
    pad_x_before = (target_x - x_dim) // 2
    pad_x_after = target_x - x_dim - pad_x_before

    # Padding for y dimension
    pad_y_before = (target_y - y_dim) // 2
    pad_y_after = target_y - y_dim - pad_y_before

    # Padding for z dimension
    pad_z_before = (target_z - z_dim) // 2
    pad_z_after = target_z - z_dim - pad_z_before

    # Apply padding
    padded_data = np.pad(data,
                         ((pad_x_before, pad_x_after),
                          (pad_y_before, pad_y_after),
                          (pad_z_before, pad_z_after)),
                         mode='constant', constant_values=0)

    return padded_data


def process_images(directory, output_directory, target_shape):
    """
    Process each image by padding each dimension to the target_shape.
    Args:
        directory (str): Directory containing input NIfTI images.
        output_directory (str): Directory to save the processed NIfTI images.
        target_shape (tuple): The target size as (x, y, z).
    """
    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Process each file in the input directory
    for filename in os.listdir(directory):
        if filename.endswith('.nii.gz'):
            input_path = os.path.join(directory, filename)
            output_filename = f"{os.path.splitext(filename)[0]}_processed.nii.gz"
            output_path = os.path.join(output_directory, output_filename)

            # Load the image
            img = nib.load(input_path)
            data = img.get_fdata()

            # Pad dimensions to target_shape
            padded_data = pad_to_target(data, target_shape)

            # Save the processed image
            padded_img = nib.Nifti1Image(padded_data, img.affine)
            nib.save(padded_img, output_path)
            print(f"Processed and saved: {output_path} with shape {padded_data.shape}")


if __name__ == '__main__':
    # Directory containing NIfTI images
    directory = 'G://semester_thesis//Project//data//data_classification//unhealthy_final'
    # Windows: 'G://semester_thesis//Project//data//data_classification//healthy_final'
    # Linux: '/home/fit_member/Documents/NS_SemesterWork/Project/data/data_classification/healthy_final'
    
    output_directory = 'G://semester_thesis//Project//data//data_classification//unhealthy_final_processed'
    # Windows: 'G://semester_thesis//Project//data//data_classification//healthy_final_processed'
    # Linux: '/home/fit_member/Documents/NS_SemesterWork/Project/data/data_classification/healthy_final_processed'
    
    # Find the maximum dimensions across all images
    max_x, max_y, max_z = find_max_dimensions(directory)

    # Define target shape based on the maximum dimensions
    target_shape = (256, 256, 256)

    # Process and pad all images to the target shape
    process_images(directory, output_directory, target_shape=target_shape)
