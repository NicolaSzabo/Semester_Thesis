import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from monai.networks.nets import DenseNet121
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, RandFlip, RandZoom
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Prepare data
data_dir = "/path/to/data"  # Replace with the actual data directory
excel_path = os.path.join(data_dir, "data_overview.xlsx")
data_overview = pd.read_excel(excel_path)

# Extract file paths, labels, and metadata
file_paths = data_overview['Nr'].apply(lambda x: os.path.join(data_dir, x)).tolist()
labels = data_overview['Classification'].tolist()
ages = data_overview['Age'].tolist()
genders = data_overview['Gender'].tolist()



# Dataset class
class HeartClassification(Dataset):
    def __init__(self, file_paths, labels, ages, genders, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.ages = ages
        self.genders = genders
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        # Load image
        img = LoadImage(image_only=True)(self.file_paths[index])
        if self.transform:
            img = self.transform(img)

        # Normalize age
        age = (self.ages[index] - min(self.ages)) / (max(self.ages) - min(self.ages))

        # One-hot encode gender
        gender = [1, 0] if self.genders[index] == 'm' else [0, 1]

        # Combine metadata
        meta = torch.tensor([age] + gender, dtype=torch.float32)

        # Label
        label = self.labels[index]

        return img, meta, label

# Define transforms
train_transform = Compose([
    EnsureChannelFirst(),
    ScaleIntensity(),
    RandFlip(spatial_axis=0, prob=0.5),
    RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
])

# Create dataset
dataset = HeartClassification(file_paths, labels, ages, genders, transform=train_transform)

# Define the multimodal model
class MultimodalDenseNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(MultimodalDenseNet, self).__init__()
        # Image branch (DenseNet121)
        self.image_branch = DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=128  # Intermediate feature size
        )

        # Metadata branch (MLP)
        self.meta_branch = torch.nn.Sequential(
            torch.nn.Linear(3, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU()
        )

        # Combined classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(128 + 32, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, img, meta):
        img_features = self.image_branch(img)
        meta_features = self.meta_branch(meta)
        combined = torch.cat((img_features, meta_features), dim=1)
        return self.classifier(combined)


# Training and validation loop
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs, writer, fold):
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, meta, labels in train_loader:
            inputs, meta, labels = inputs.to(device), meta.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, meta)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data)
            total_train += labels.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_train.double() / total_train
        print(f"Fold {fold + 1}, Epoch {epoch + 1}: Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}")
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, meta, labels in val_loader:
                inputs, meta, labels = inputs.to(device), meta.to(device), labels.to(device)
                outputs = model(inputs, meta)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data)
                total_val += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = correct_val.double() / total_val
        print(f"Fold {fold + 1}, Epoch {epoch + 1}: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

    return val_loss, val_acc

# K-Fold Cross Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=0)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    print(f"Fold {fold + 1}/5")
    log_dir = f"./runs/fold_{fold + 1}"
    writer = SummaryWriter(log_dir=log_dir)

    train_subsampler = SubsetRandomSampler(train_idx)
    val_subsampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(dataset, batch_size=4, sampler=train_subsampler, num_workers=2)
    val_loader = DataLoader(dataset, batch_size=4, sampler=val_subsampler, num_workers=2)

    model = MultimodalDenseNet(num_classes=2).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    val_loss, val_acc = train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs=10, writer=writer, fold=fold)
    fold_results.append({'fold': fold, 'val_loss': val_loss, 'val_acc': val_acc})

    writer.close()

# Average metrics across folds
avg_val_loss = np.mean([result['val_loss'] for result in fold_results])
avg_val_acc = np.mean([result['val_acc'].cpu().item() for result in fold_results])
print(f"Average Validation Loss: {avg_val_loss:.4f}, Average Validation Accuracy: {avg_val_acc:.4f}")
