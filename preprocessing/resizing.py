import os
import nibabel as nib
import numpy as np

def find_max_dimensions(directory):
    """
    Find the maximum x, y, and z dimensions across all NIfTI images in a directory.
    """
    max_x, max_y, max_z = 0, 0, 0
    for filename in os.listdir(directory):
        if filename.endswith('.nii.gz'):
            file_path = os.path.join(directory, filename)
            img = nib.load(file_path)
            data = img.get_fdata()
            x_dim, y_dim, z_dim = data.shape
            max_x = max(max_x, x_dim)
            max_y = max(max_y, y_dim)
            max_z = max(max_z, z_dim)
            print(f"{filename} has dimensions: {data.shape}")
    print(f"Max dimensions: x={max_x}, y={max_y}, z={max_z}")
    return max_x, max_y, max_z



def pad_to_target(data, target_shape):
    """
    Pad the x, y, and z dimensions of a 3D numpy array with NaN to match target_shape.
    """
    x_dim, y_dim, z_dim = data.shape
    target_x, target_y, target_z = target_shape

    # Calculate padding for each dimension
    pad_x_before = (target_x - x_dim) // 2
    pad_x_after = target_x - x_dim - pad_x_before
    pad_y_before = (target_y - y_dim) // 2
    pad_y_after = target_y - y_dim - pad_y_before
    pad_z_before = (target_z - z_dim) // 2
    pad_z_after = target_z - z_dim - pad_z_before

    # Pad the array with NaN
    padded_data = np.pad(data,
                         ((pad_x_before, pad_x_after),
                          (pad_y_before, pad_y_after),
                          (pad_z_before, pad_z_after)),
                         mode='constant', constant_values=np.nan)
    return padded_data

def process_images(directory, output_directory, target_shape):
    """
    Process each image by padding each dimension with NaN to the target_shape.
    Skip images larger than target_shape and print them at the end.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    skipped_files = []  # To store filenames that are too large

    for filename in os.listdir(directory):
        if filename.endswith('.nii.gz'):
            input_path = os.path.join(directory, filename)
            output_filename = f"{os.path.splitext(filename)[0]}.nii.gz"
            output_path = os.path.join(output_directory, output_filename)

            # Load the image
            img = nib.load(input_path)
            data = img.get_fdata()
            x_dim, y_dim, z_dim = data.shape

            # Check if the image is too large
            if x_dim > target_shape[0] or y_dim > target_shape[1] or z_dim > target_shape[2]:
                print(f"Skipping {filename}: dimensions {data.shape} exceed target shape {target_shape}.")
                skipped_files.append(filename)
                continue

            # Skip images that already match the target shape
            if data.shape == target_shape:
                print(f"{filename} already matches target shape {target_shape}, skipping.")
                continue

            # Pad the image to the target shape with NaN
            padded_data = pad_to_target(data, target_shape)

            # Save the processed image
            padded_img = nib.Nifti1Image(padded_data, img.affine)
            nib.save(padded_img, output_path)
            print(f"Processed and saved: {output_path} with shape {padded_data.shape}")

    # Print all skipped files at the end
    print("\nFiles skipped because they are larger than the target shape:")
    for skipped_file in skipped_files:
        print(skipped_file)

if __name__ == '__main__':
    # Directory containing NIfTI images
    directory = "G://data//preprocessing//masked_heart_specific"
    output_directory = "G://data//preprocessing//resized_heart_specific"

    # Find the maximum dimensions across all images
    max_x, max_y, max_z = find_max_dimensions(directory)

    # Define target shape based on the maximum dimensions or fixed shape
    target_shape = (256, 256, 256)  # Use (max_x, max_y, max_z) if dynamic target shape is desired

    # Process and pad all images to the target shape
    process_images(directory, output_directory, target_shape)
