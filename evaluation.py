import os
import torch
import nibabel as nib
import numpy as np
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, Resize
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
from monai.networks.nets import DenseNet121
from monai.metrics import ROCAUCMetric
from omegaconf import OmegaConf

# Specify the paths to the configuration and model files
config_path = '/home/fit_member/Documents/NS_SemesterWork/config.yaml'
model_path = '/home/fit_member/Documents/NS_SemesterWork/best_model.pth'  # Replace with the actual filename

# Load configuration
config = OmegaConf.load(config_path)

# Define transforms (should match the validation transforms used during training)
val_transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity(),
    Resize((128, 128, 128))
])

# Define the evaluation dataset class
class HeartClassification(Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]

# Load the test data 
test_x = ...  # List of test image file paths
test_y = ...  # List of test labels
test_ds = HeartClassification(test_x, test_y, val_transforms)
test_loader = DataLoader(test_ds, batch_size=config.dataset.batch_size, num_workers=config.dataset.num_workers)

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=len(set(test_y))).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Evaluation function
def evaluate_model(model, test_loader, device):
    y_true, y_pred = [], []
    auc_metric = ROCAUCMetric()
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Collect predictions and true labels
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Compute classification report
    report = classification_report(y_true, y_pred, target_names=config.dataset.class_names)
    print("Classification Report:\n", report)
    
    # Calculate and print AUC
    auc = auc_metric(torch.tensor(y_pred), torch.tensor(y_true))
    print(f"AUC: {auc:.4f}")
    return report, auc

# Run evaluation
evaluate_model(model, test_loader, device)
