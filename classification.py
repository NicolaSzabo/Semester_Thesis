import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import nibabel as nib
import PIL
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import classification_report
import monai.transforms as mt
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)
from monai.utils import set_determinism

#print_config()



### Set deterministic training for reproducibility
set_determinism(seed = 0)


### Read image filenames from the dataset folders

data_dir = '/home/fit_member/Documents/NS_SemesterWork/data/data_classification'

class_names = sorted(x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x)))
num_class = len(class_names)
print(num_class)

image_files = [
    [os.path.join(data_dir, class_names[i], x) for x in os.listdir(os.path.join(data_dir, class_names[i]))]
    for i in range(num_class)
]
num_each = [len(image_files[i]) for i in range(num_class)]
image_files_list = []
image_class = []
for i in range(num_class):
    image_files_list.extend(image_files[i])
    image_class.extend([i] * num_each[i])
num_total = len(image_class)

# Use LoadImage from MONAI to transform NIfTI files as PyTorch tensors
first_image = LoadImage()(image_files_list[0]) # LoadImage returns (image, metadata), so [0] accesses the image
image_shape = first_image.shape


print(f"Total image count: {num_total}")
print(f"Image dimensions: {image_shape}")
print(f"Label names: {class_names}")
print(f"Label counts: {num_each}")


### Randomly pick images from the dataset to visualize and check

plt.subplots(3, 3, figsize = (8, 8))
for i, k in enumerate(np.random.randint(num_total, size = 9)):
    im = nib.load(image_files_list[k])
    data = im.get_fdata()
    arr = data[:, :, data.shape[2] // 2]
    plt.subplot(3, 3, i + 1)
    plt.xlabel(class_names[image_class[k]])
    plt.imshow(arr, cmap = 'gray')
plt.tight_layout()
plt.show()


### Prepare training, validation and test data lists
