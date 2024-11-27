import os
import torch
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
from monai.networks.nets import DenseNet121
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, RandFlip, RandZoom
from monai.utils import set_determinism
from omegaconf import OmegaConf
from datetime import datetime

# Check CUDA availability
print(torch.version.cuda)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f'GPU is available. Device in use: {torch.cuda.get_device_name(0)}')
else:
    print('No GPU available. Using CPU instead.')

# Set deterministic training
set_determinism(seed=0)

# Configuration and paths
config = OmegaConf.load('/home/fit_member/Documents/NS_SemesterWork/Project/config_final_models.yaml')
print(OmegaConf.to_yaml(config))

start_time = datetime.now()
start_time_str = start_time.strftime('%Y-%d-%m_%H-%M')

# Log results function
def log_results(config, start_time, end_time, duration, model_filename, file_path):
    results = {
        'run_id': f"Run_{datetime.now().strftime('%Y-%d-%m_%H-%M')}",
        'start_time': start_time,
        'end_time': end_time,
        'duration': duration,
        'model_filename': model_filename,
        'config': config,
    }
    with open(file_path, 'a') as f:
        OmegaConf.save(config=OmegaConf.create(results), f=f)

# Directories
base_results_dir = '/home/fit_member/Documents/NS_SemesterWork/Project/results/final_models'
run_dir = os.path.join(base_results_dir, f"run_{start_time.strftime('%Y-%d-%m_%H-%M')}")
os.makedirs(run_dir, exist_ok=True)

# Data preparation
data_dir = config.dataset.data_dir
class_names = sorted(x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x)))
num_class = len(class_names)

image_files = [
    [os.path.join(data_dir, class_names[i], x) for x in os.listdir(os.path.join(data_dir, class_names[i]))]
    for i in range(num_class)
]

num_each = [len(image_files[i]) for i in range(num_class)]
X = [file for class_files in image_files for file in class_files]
y = [i for i in range(num_class) for _ in range(num_each[i])]

# Transforms
train_transform = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity(),
    RandFlip(spatial_axis=0, prob=0.5),
    RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
])

# Dataset
class HeartClassification(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        img = self.transform(self.X[index])
        label = self.y[index]
        return img, label

dataset = HeartClassification(X, y, transform=train_transform)

# Split into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = config.dataset.batch_size
num_workers = config.dataset.num_workers

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Model, loss, and optimizer
model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=num_class).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)

# TensorBoard setup
log_dir = f"/home/fit_member/Documents/NS_SemesterWork/Project/runs/final_models/experiment_{start_time.strftime('%Y-%d-%m_%H-%M')}"
writer = SummaryWriter(log_dir=log_dir)

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data)
            total_train += labels.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_train.double() / total_train
        print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}")
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data)
                total_val += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = correct_val.double() / total_val
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

    return model

# Train the model
epochs = config.training.epochs
trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device)

# Save the model
model_filename = f"final_model_{datetime.now().strftime('%Y-%d-%m_%H-%M')}.pth"
torch.save(trained_model.state_dict(), os.path.join(run_dir, model_filename))

# Log results
end_time = datetime.now()
log_results(config=config, start_time=start_time_str, end_time=end_time.strftime('%Y-%d-%m_%H-%M'), duration=str(end_time - start_time), model_filename=model_filename, file_path=os.path.join(base_results_dir, "results_final_models.yaml"))

writer.close()
