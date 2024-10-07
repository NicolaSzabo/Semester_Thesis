# First file I created just to play around with the NIFTI files and plotting them. Not really important.


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
import nilearn as nil


#Check data for being DICOM or not

path_CT = 'cropped_images/16-1217_5_0_B31S.nii.gz'
path_mask = 'masks/heart_16-1217.nii.gz'


# Load NIFTI file
CT_img = nib.load(path_CT)
mask_img = nib.load(path_mask)

# Get voxel dimensions (voxel spacing)
voxel_dims = CT_img.header.get_zooms()  # Returns (x_spacing, y_spacing, z_spacing)

# Access data from the NIfTI file
CT_data = CT_img.get_fdata()
mask_data = mask_img.get_fdata()


# Plot a series of slices with overlayed mask and corrected voxel spacing
fig_rows = 4
fig_cols = 4
n_subplots = fig_rows * fig_cols
n_slice = CT_data.shape[0]
step_size = n_slice // n_subplots
plot_range = n_subplots * step_size
start_stop = int((n_slice - plot_range) / 2)

fig, axs = plt.subplots(fig_rows, fig_cols, figsize=[10, 10])

for idx, img in enumerate(range(start_stop, plot_range, step_size)):
    axs.flat[idx].imshow(np.rot90(CT_data[:, img, :]), cmap = 'gray', aspect = voxel_dims[2] / voxel_dims[0])  # Correct aspect ratio
    axs.flat[idx].imshow(np.rot90(mask_data[:, img, :]), cmap = 'Reds', alpha = 0.3, aspect=voxel_dims[2] / voxel_dims[0])  # Overlay mask
    
    axs.flat[idx].axis('off')

plt.tight_layout()
plt.show()



# Visualize a single slice with voxel spacing corrected
slice_idx = 180  # Select a slice index

plt.figure(figsize=[6, 6])

# Set aspect ratio using the voxel dimensions
plt.imshow(np.rot90(CT_data[:, slice_idx, :]), cmap = 'gray', aspect = voxel_dims[2] / voxel_dims[0])  # Adjust aspect ratio
plt.imshow(np.rot90(mask_data[:, slice_idx, :]), cmap = 'Reds', alpha = 0.3, aspect=voxel_dims[2] / voxel_dims[0])  # Overlay mask

plt.axis('off')  # Hide axes
plt.title(f"Single Slice with Mask Overlay - Slice {slice_idx}")
plt.show()