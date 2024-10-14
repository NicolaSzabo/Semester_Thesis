# data_loader.py
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class CTDataset(Dataset):
    def __init__(self, data_path, transform=None):
        # Load your data here
        self.data = ...  # Load from NIFTI files or any other format
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img

def get_dataloaders(train_dir, batch_size=16, shuffle=True):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = CTDataset(train_dir, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    return train_loader
