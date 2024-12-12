import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from monai.networks.nets import DenseNet121, DenseNet169
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, RandFlip, RandZoom, Resize, RandGaussianNoise
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from omegaconf import OmegaConf
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision.models import DenseNet

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration and paths
config = OmegaConf.load('/home/fit_member/Documents/NS_SemesterWork/Project/config_cross_validation.yaml')
# DIRECTORY:
# Linux: '/home/fit_member/Documents/NS_SemesterWork/Project/config.yaml'
# Windows: 'C://Users//nicol//OneDrive//Desktop//semester_thesis//Project//config.yaml'
print(OmegaConf.to_yaml(config))

start_time = datetime.now()
start_time_str = start_time.strftime('%Y-%d-%m_%H-%M')

def log_results(config, start_time, end_time, duration, file_path = '/home/fit_member/Documents/NS_SemesterWork/Project/results/cross_validation/results_cross_validation.yaml'):
    # DIRECTORY:
    # Linux: '/home/fit_member/Documents/NS_SemesterWork/Project/results/results_log.yaml'
    # Windows: 'C://Users//nicol//OneDrive//Desktop//semester_thesis//Project//results//results_log.yaml'
    timestamp = datetime.now().strftime('%Y-%d-%m_%H-%M')
    results = {
        'run_id': f"Run_{timestamp}",
        'start_time': start_time,
        'end_time': end_time,
        'duration': duration,
        'config': config,
    }
    with open(file_path, 'a') as f:
        OmegaConf.save(config = OmegaConf.create(results), f = f)






base_results_dir = '/home/fit_member/Documents/NS_SemesterWork/Project/results/cross_validation'
# DIRECTORY:
# Linux: '/home/fit_member/Documents/NS_SemesterWork/Project/results'
# Windows: 'C://Users//nicol//OneDrive//Desktop//semester_thesis//Project//results'
run_dir = os.path.join(base_results_dir, f"run_{start_time.strftime('%Y-%d-%m_%H-%M')}")
os.makedirs(run_dir, exist_ok=True)

data_dir = config.dataset.data_dir



# Update Dataset to Include Volumes
class HeartClassification(Dataset):
    LABEL_MAP = {'healthy': 0, 'pathological': 1}

    def __init__(self, file_paths, labels, volumes, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.volumes = volumes  # Volume column as metadata
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        # Load image
        img = LoadImage(image_only=True)(self.file_paths[index])
        if self.transform:
            img = self.transform(img)

        # Normalize volume
        volume = (self.volumes[index] - min(self.volumes)) / (max(self.volumes) - min(self.volumes))
        meta = torch.tensor([volume], dtype=torch.float32)

        # Label
        label = self.LABEL_MAP[self.labels[index]]
        label = torch.tensor(label, dtype=torch.long)

        return img, meta, label



# Define transforms
train_transform = Compose([
    EnsureChannelFirst(),
    ScaleIntensity(),
    RandFlip(spatial_axis=0, prob=0.5),
    RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
    RandGaussianNoise(prob = 0.5, mean = 0.0, std = 0.1),
])


# Updated Multimodal Model
class MultimodalDenseNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(MultimodalDenseNet, self).__init__()
        # Image branch
        self.image_branch = DenseNet121(
            spatial_dims=3,
            dropout_prob=0.5,
            in_channels=1,
            out_channels=128  # Intermediate feature size
        )

        # Metadata branch (1 input for volume)
        self.meta_branch = torch.nn.Sequential(
            torch.nn.Linear(1, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 16),
            torch.nn.ReLU()
        )

        # Combined classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(128 + 16, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, img, meta):
        img_features = self.image_branch(img)
        meta_features = self.meta_branch(meta)
        combined = torch.cat((img_features, meta_features), dim=1)
        return self.classifier(combined)


# Load the Excel file and extract volumes
excel_data = pd.read_excel('/home/fit_member/Documents/NS_SemesterWork/Project/data/data_overview_binary_cleaned_256.xlsx')

# Filter for good quality and extract relevant columns
good_quality_data = excel_data[excel_data['quality'] == 'good']
file_paths = good_quality_data['Nr'].apply(lambda x: os.path.join(data_dir, f"{x}.nii.gz")).tolist()
labels = good_quality_data['Classification'].tolist()
volumes = good_quality_data['Volume_mL'].tolist()  # Use precomputed volumes

num_class = len(set(labels))
class_names = sorted(set(labels))
class_counts = pd.Series(labels).value_counts()


print(f"Number of Classes: {num_class}")
print(f"Number of files per class: {class_counts}")
print(f"Total samples: {len(file_paths)}")
print(f"Sample file path: {file_paths[0]}, Label: {labels[0]}")


# Dataset and DataLoader
dataset = HeartClassification(file_paths, labels, volumes, transform=train_transform)
k_folds = config.training.k_folds
batch_size = config.dataset.batch_size
num_workers = config.dataset.num_workers





# Training and validation loop
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, meta, labels in train_loader:
            labels = labels[0] if isinstance(labels, tuple) else labels

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
        print(f"Epoch {epoch + 1}: Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}")
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
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

    return val_loss, val_acc




misclassified_data = []
# Function to compute metrics for evaluation
def evaluate_model(model, val_loader, device, file_paths, labels, fold):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    # Validation indices
    val_indices = list(val_loader.sampler.indices)
    X_val = [file_paths[i] for i in val_indices]  # Validation file paths
    y_val = [labels[i] for i in val_indices]  # Validation labels

    # Convert `y_val` to integers
    y_val = [dataset.LABEL_MAP[label] for label in y_val]

    with torch.no_grad():
        for inputs, meta, labels in val_loader:
            # Move all inputs to the same device
            inputs = inputs.to(device)
            meta = meta.to(device)  # Explicitly move metadata
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs, meta)

            # Compute predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            # Collect results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Misclassified samples
    for i, pred in enumerate(all_preds):
        actual_label = y_val[i]
        file_path = X_val[i]
        if pred != actual_label:
            misclassified_data.append({
                'fold': fold,
                'file_path': file_path,
                'actual_label': actual_label,
                'predicted_label': pred
            })

    return np.array(y_val), all_preds, np.array(all_probs)


# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names, fold):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix (Fold {fold + 1})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(run_dir, f"confusion_matrix_fold_{fold}.png"))
    plt.close()


# Function to plot ROC and PRC curves
def plot_roc_prc(y_true, y_probs, class_names, fold):
    for i, class_name in enumerate(class_names):
        # Create binary labels for the current class
        binary_y_true = (y_true == i).astype(int)

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(binary_y_true, y_probs[:, i])
        roc_auc = auc(fpr, tpr)

        # Plot ROC
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')  # Random baseline
        plt.title(f'ROC Curve - {class_name} (Fold {fold + 1})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(run_dir, f"roc_{class_name}_fold_{fold}.png"))
        plt.close()

        # Compute PRC
        precision, recall, _ = precision_recall_curve(binary_y_true, y_probs[:, i])
        pr_auc = auc(recall, precision)

        # Plot PRC
        no_skill = np.sum(binary_y_true) / len(binary_y_true)
        plt.figure()
        plt.plot(recall, precision, label=f'PRC curve (AUC = {pr_auc:.2f})')
        plt.title(f'Precision-Recall Curve - {class_name} (Fold {fold + 1})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc='lower left')
        plt.savefig(os.path.join(run_dir, f"prc_{class_name}_fold_{fold}.png"))
        plt.close()




# K-Fold Cross Validation
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=0)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    print(f"Fold {fold + 1}/{k_folds}")

    log_dir = f"/home/fit_member/Documents/NS_SemesterWork/Project/runs/cross_validation/experiment_{start_time.strftime('%Y-%d-%m_%H-%M')}/fold_{fold + 1}"
    writer = SummaryWriter(log_dir=log_dir)

    train_subsampler = SubsetRandomSampler(train_idx)
    val_subsampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler, num_workers=num_workers)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler, num_workers=num_workers)

    model = MultimodalDenseNet(num_classes=2).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = config.training.lr)


    val_loss, val_acc = train_and_validate(model, train_loader, val_loader, criterion, optimizer, config.training.epochs)
    fold_results.append({'fold': fold, 'val_loss': val_loss, 'val_acc': val_acc})
    torch.save(model.state_dict(), os.path.join(run_dir, f"model_fold_{fold}.pth"))

    # Evaluate the model and compute metrics
    y_true, y_pred, y_probs = evaluate_model(
        model=model,
        val_loader=val_loader,
        device=device,
        file_paths=file_paths,
        labels=labels,
        fold=fold
    )

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names, fold)

    # Plot ROC and PRC curves
    plot_roc_prc(y_true, y_probs, class_names, fold)

    # Save misclassified data to a CSV file
    misclassified_df = pd.DataFrame(misclassified_data)
    csv_path = os.path.join(run_dir, "misclassified_files.csv")
    misclassified_df.to_csv(csv_path, index=False)

    print(f"Misclassified files saved to {csv_path}")

    writer.close()

# Average metrics across folds
avg_val_loss = np.mean([result['val_loss'] for result in fold_results])
avg_val_acc = np.mean([result['val_acc'].cpu().item() for result in fold_results])
print(f"Average Validation Loss: {avg_val_loss:.4f}, Average Validation Accuracy: {avg_val_acc:.4f}")


end_time = datetime.now()
end_time_str = end_time.strftime('%Y-%d-%m_%H-%M')
duration = end_time - start_time
duration_str = str(duration)


# Log results to YAML
log_results(config = config,
            start_time = start_time_str,
            end_time = end_time_str,
            duration = duration_str)
