import os
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from monai.data import decollate_batch
from monai.networks.nets import DenseNet121
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, RandFlip, RandZoom
from monai.utils import set_determinism
from omegaconf import OmegaConf
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc
import seaborn as sns


# Check CUDA availability
print(torch.version.cuda)
if torch.cuda.is_available():
    print('GPU is available. Device in use: ')
    print(torch.cuda.get_device_name(0))
else:
    print('No GPU available. Using CPU instead.')

# Set deterministic training
set_determinism(seed=0)

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



# Prepare data
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
for i, (path, label) in enumerate(zip(X, y)):
    print(f"Index: {i}, Path: {path}, Label: {label}")


# Verify data
first_image = LoadImage()(X[0])
print(f"Total image count: {len(X)}, Image dimensions: {first_image.shape}, Label names: {class_names}, Label counts: {num_each}")

# Dataset and transforms
train_transform = Compose([
    LoadImage(image_only = True),
    EnsureChannelFirst(),
    ScaleIntensity(),
    RandFlip(spatial_axis=0, prob=0.5),
    RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
])



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



# Data preparation for K-Fold
dataset = HeartClassification(X, y, transform = train_transform)
k_folds = config.training.k_folds
batch_size = config.dataset.batch_size
num_workers = config.dataset.num_workers
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





# Training and validation
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs, device):
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
        print(f"Epoch {epoch + 1}: Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}")
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

    return val_loss, val_acc


import pandas as pd

# Create an empty list to store misclassified data across all folds
misclassified_data = []


# Function to compute metrics for evaluation
def evaluate_model(model, val_loader, device, X_val, fold):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    # Validation indices
    val_indices = list(val_loader.sampler.indices)
    X_val = [X[i] for i in val_indices]  # Validation file paths
    y_val = [y[i] for i in val_indices]  # Validation labels

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Get predictions
            probs = torch.softmax(outputs, dim=1)  # Probabilities for each class
            preds = torch.argmax(probs, dim=1)

            # Append predictions and true labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert lists to numpy arrays for easier manipulation
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    y_true = y_val
    y_true = np.array(y_true)# Ground truth labels
    y_pred = all_preds  # Predictions from the model

    # Identify misclassified samples and save to misclassified_data
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

    return y_true, y_pred, np.array(all_probs)


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
        plt.figure()
        plt.plot(recall, precision, label=f'PRC curve (AUC = {pr_auc:.2f})')
        plt.title(f'Precision-Recall Curve - {class_name} (Fold {fold + 1})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc='lower left')
        plt.savefig(os.path.join(run_dir, f"prc_{class_name}_fold_{fold}.png"))
        plt.close()









# K-Fold Cross Validation
kfold = KFold(n_splits = k_folds, shuffle=True)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    print(f"Fold {fold + 1}/{k_folds}")
    torch.cuda.empty_cache()

    log_dir = f"/home/fit_member/Documents/NS_SemesterWork/Project/runs/cross_validation/experiment_{start_time.strftime('%Y-%d-%m_%H-%M')}/fold_{fold + 1}"

    # DIRECTORY:
    # Linux: f"/home/fit_member/Documents/NS_SemesterWork/Project/runs/experiment_{start_time.strftime('%Y-%d-%m_%H-%M')}/fold_{fold + 1}"
    # Windows: r"C:\Users\nicol\OneDrive\Desktop\semester_thesis\Project\runs\experiment_{start_time.strftime('%Y-%d-%m_%H-%M')}\fold_{fold + 1}"
    writer = SummaryWriter(log_dir=log_dir)

    train_subsampler = SubsetRandomSampler(train_idx)
    val_subsampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(dataset, batch_size = batch_size, sampler = train_subsampler, num_workers = num_workers)
    val_loader = DataLoader(dataset, batch_size = batch_size, sampler = val_subsampler, num_workers = num_workers)

    model = DenseNet121(spatial_dims = 3, in_channels = 1, out_channels = num_class).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)

    val_loss, val_acc = train_and_validate(model, train_loader, val_loader, criterion, optimizer, config.training.epochs, device)
    fold_results.append({'fold': fold, 'val_loss': val_loss, 'val_acc': val_acc})
    torch.save(model.state_dict(), os.path.join(run_dir, f"model_fold_{fold}.pth"))

    # Evaluate and compute metrics
    y_true, y_pred, y_probs = evaluate_model(model, val_loader, device, X, fold)

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names, fold)

    # Plot ROC and PRC curves
    plot_roc_prc(y_true, y_probs, class_names, fold)

    # Convert misclassified data to a DataFrame
    misclassified_df = pd.DataFrame(misclassified_data)

    # Save to CSV
    csv_path = os.path.join(run_dir, "misclassified_files.csv")
    misclassified_df.to_csv(csv_path, index=False)

    print(f"Misclassified files saved to {csv_path}")

    writer.close()




avg_val_acc = np.mean([result['val_acc'].cpu().item() for result in fold_results])
avg_val_loss = np.mean([result['val_loss'] for result in fold_results])
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

