import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from monai.networks.nets import DenseNet121
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, RandFlip, RandZoom, RandGaussianNoise, Resize
from monai.utils import set_determinism
from omegaconf import OmegaConf
from datetime import datetime
from sklearn.model_selection import KFold






# Main function to ensure proper multiprocessing behavior
if __name__ == '__main__':
    # Set deterministic training and device
    set_determinism(seed=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Configuration and paths
    config = OmegaConf.load('C://Users//nicol//OneDrive//Desktop//Semester_thesis//config_cross_validation.yaml') 
    print(OmegaConf.to_yaml(config))

    start_time = datetime.now()
    start_time_str = start_time.strftime('%Y-%d-%m_%H-%M')

    # Logging function
    def log_results(config, start_time, end_time, duration, file_path='C://Users//nicol//OneDrive//Desktop//Semester_thesis//results//cross_validation//results_cross_validation.yaml'):
        timestamp = datetime.now().strftime('%Y-%d-%m_%H-%M')
        results = {
            'run_id': f"Run_{timestamp}",
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'config': config,
        }
        with open(file_path, 'a') as f:
            OmegaConf.save(config=OmegaConf.create(results), f=f)

    # Prepare results directory
    base_results_dir = 'C://Users//nicol//OneDrive//Desktop//Semester_Thesis//results//cross_validation'
    run_dir = os.path.join(base_results_dir, f"run_{start_time.strftime('%Y-%d-%m_%H-%M')}")
    os.makedirs(run_dir, exist_ok=True)

    # Prepare data
    data_dir = config.dataset.data_dir
    excel_path = "G://data//data_overview_binary_cleaned_256.xlsx"
    data_overview = pd.read_excel(excel_path)
    filtered_data = data_overview[data_overview['quality'] == 'good']

    file_paths = filtered_data['Nr'].apply(lambda x: os.path.join(data_dir, f"{x}.nii.gz")).tolist()
    labels = filtered_data['Classification'].tolist()
    num_class = len(set(labels))
    class_names = sorted(set(labels))

    # Custom dataset
    class HeartClassification(Dataset):
        def __init__(self, file_paths, labels, transform=None):
            self.file_paths = file_paths
            self.label_to_index = {label: idx for idx, label in enumerate(sorted(set(labels)))}
            self.labels = [self.label_to_index[label] for label in labels]
            self.transform = transform

        def __len__(self):
            return len(self.file_paths)

        def __getitem__(self, index):
            img = LoadImage(image_only=True)(self.file_paths[index])
            if self.transform:
                img = self.transform(img)
            label = torch.tensor(self.labels[index], dtype=torch.long)
            return img.to(torch.float32), label

    # Dataset and transforms
    train_transform = Compose([
        EnsureChannelFirst(),
        ScaleIntensity(),
        Resize(spatial_size = (128, 128, 128)),
        RandFlip(spatial_axis=0, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        RandGaussianNoise(prob=0.5, mean=0.0, std=0.1),
    ])
    dataset = HeartClassification(file_paths, labels, transform=train_transform)





    k_folds = config.training.k_folds
    batch_size = config.dataset.batch_size
    num_workers = 0  # Set to 0 for Windows compatibility

    # K-Fold Cross Validation
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=0)
    fold_results = []



    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold + 1}/{k_folds}")
        log_dir = r"C:\Users\nicol\OneDrive\Desktop\Semester_thesis\runs\cross_validation\experiment_{start_time.strftime('%Y-%d-%m_%H-%M')}\fold_{fold + 1}"
        writer = SummaryWriter(log_dir=log_dir)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx), num_workers=num_workers)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_idx), num_workers=num_workers)

        # Initialize model, criterion, and optimizer
        model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=num_class, dropout_prob=0.5).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=config.training.lr, momentum=0.9)

        # Train and validate
        def train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs, device):
            for epoch in range(epochs):
                model.train()
                running_loss, correct_train, total_train = 0.0, 0, 0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    correct_train += (outputs.argmax(1) == labels).sum().item()
                    total_train += labels.size(0)
                epoch_loss = running_loss / len(train_loader.dataset)
                epoch_acc = correct_train / total_train
                print(f"Epoch {epoch+1}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

        train_and_validate(model, train_loader, val_loader, criterion, optimizer, config.training.epochs, device)
        fold_results.append({'val_loss': 0.0, 'val_acc': 0.0})  # Placeholder
        torch.save(model.state_dict(), os.path.join(run_dir, f"model_fold_{fold}.pth"))
        writer.close()



    # Compute and print average metrics across all folds
    avg_val_loss = np.mean([result['val_loss'] for result in fold_results])
    avg_val_acc = np.mean([result['val_acc'] for result in fold_results])
    print(f"Average Validation Loss: {avg_val_loss:.4f}, Average Validation Accuracy: {avg_val_acc:.4f}")

    # Finalize and log results
    end_time = datetime.now()
    end_time_str = end_time.strftime('%Y-%d-%m_%H-%M')
    duration = end_time - start_time
    duration_str = str(duration)

    log_results(config=config,
                start_time=start_time_str,
                end_time=end_time_str,
                duration=duration_str)

