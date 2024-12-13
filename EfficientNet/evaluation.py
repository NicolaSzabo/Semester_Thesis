import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from efficientnet_pytorch import EfficientNet
import os
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, RandFlip, RandZoom, Resize, \
    RandGaussianNoise
from torch.utils.data import Dataset
import nibabel as nib




class HeartClassification(Dataset):
    LABEL_MAP = {'healthy': 0, 'pathological': 1}

    def __init__(self, file_paths, labels, features_df, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.features_df = features_df  # DataFrame with metadata features
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        # Load 3D NIfTI image
        img = nib.load(self.file_paths[index]).get_fdata()

        # Extract middle slices for each view
        middle_axial = img[:, :, img.shape[2] // 2]  # Axial view
        middle_coronal = img[:, img.shape[1] // 2, :]  # Coronal view
        middle_sagittal = img[img.shape[0] // 2, :, :]  # Sagittal view

        # Stack slices into 3 channels
        multi_view_img = np.stack([middle_coronal, middle_axial, middle_sagittal], axis=0)

        # Apply transforms
        if self.transform:
            multi_view_img = self.transform(multi_view_img)

        # Ensure the output is a torch.Tensor
        if isinstance(multi_view_img, torch.Tensor):
            multi_view_img = multi_view_img.float()
        elif isinstance(multi_view_img, np.ndarray):
            multi_view_img = torch.from_numpy(multi_view_img).float()
        else:  # Handle MONAI MetaTensor
            multi_view_img = torch.from_numpy(np.array(multi_view_img)).float()

        # Fetch metadata features
        meta_features = torch.tensor(self.features_df.iloc[index].values, dtype=torch.float32)

        # Label
        label = self.LABEL_MAP[self.labels[index]]
        label = torch.tensor(label, dtype=torch.long)

        return multi_view_img, meta_features, label


# Dataset and transforms
train_transform = Compose([
    ScaleIntensity(),
    Resize(spatial_size=(224, 224)),  # Resize to EfficientNet's input size
    RandFlip(prob=0.5, spatial_axis=1),  # Flip horizontally
    RandGaussianNoise(prob=0.5, mean=0.0, std=0.1),
])






# --------------------------
# CONFIGURATION
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/home/fit_member/Documents/NS_SemesterWork/Project/results/cross_validation/run_2024-13-12_10-57/model_fold_3.pth"  # Update with your trained model path
test_data_path = '/home/fit_member/Documents/NS_SemesterWork/Project/data/data_overview_binary_cleaned_256.xlsx'  # Path to test set
data_dir = '/home/fit_member/Documents/NS_SemesterWork/Project/data_final_without_aorta/'  # Path to NIfTI files
batch_size = 8


# --------------------------
# LOAD TEST DATA
# --------------------------
import pandas as pd

# Load test data from your Excel file
test_data = pd.read_excel(test_data_path)
good_data = test_data[test_data['quality'] == 'good']
filtered_data = good_data[test_data['data_without_aorta'] == 'yes']

file_paths = filtered_data['Nr'].apply(lambda x: os.path.join(data_dir, f"{x}.nii.gz")).tolist()
labels = filtered_data['Classification'].tolist()

features_df = filtered_data[['Volume_mL', 'Mean_Intensity', 'Std_Intensity', 'Min_Intensity', 'Compactness', 'Surface_mm2']]

# Initialize test dataset and DataLoader
test_dataset = HeartClassification(file_paths, labels, features_df, transform=train_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)





# --------------------------
# LOAD TRAINED MODEL
# --------------------------
# Define MultimodalEfficientNet (copy-paste the class definition)
class MultimodalEfficientNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(MultimodalEfficientNet, self).__init__()

        # EfficientNet Feature-Extractor
        self.image_branch = EfficientNet.from_pretrained("efficientnet-b0", in_channels=3)
        self.image_branch._fc = torch.nn.Identity()  # Remove final FC layer
        self.image_fc = torch.nn.Linear(1280, 128)   # Reduce to 128 features

        # Metadata branch
        self.meta_branch = torch.nn.Sequential(
            torch.nn.Linear(6, 8),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(8),
            torch.nn.Linear(8, 16),
            torch.nn.ReLU()
        )

        # Combined classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(128 + 16, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.7),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, img, meta):
        img_features = self.image_branch(img)
        img_features = self.image_fc(img_features)

        meta_features = self.meta_branch(meta)

        combined = torch.cat((img_features, meta_features), dim=1)
        return self.classifier(combined)

# Initialize the model
model = MultimodalEfficientNet(num_classes=2).to(device)

# Load the trained weights
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()




# --------------------------
# COMPUTE METRICS
# --------------------------
def evaluate_model(model, dataloader, device):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, meta, labels in dataloader:  # Unpack inputs, metadata, and labels
            inputs, meta, labels = inputs.to(device), meta.to(device), labels.to(device)

            # Forward pass (pass both image and metadata to the model)
            outputs = model(inputs, meta)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["Healthy", "Pathological"])

    return accuracy, cm, report, all_labels, all_preds

accuracy, cm, report, y_true, y_pred = evaluate_model(model, test_loader, device)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Healthy", "Pathological"],
            yticklabels=["Healthy", "Pathological"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --------------------------
# SALIENCY MAP FUNCTION
# --------------------------
def compute_saliency_map(model, input_image, target_class, device):
    input_image = input_image.unsqueeze(0).to(device).requires_grad_()  # Add batch dimension
    dummy_meta = torch.zeros((1, 6), device=device).requires_grad_()    # Dummy metadata with 6 features

    # Forward pass
    output = model(input_image, dummy_meta)  # Pass both image and dummy metadata
    score = output[0, target_class]  # Score for the target class

    # Backward pass to compute gradients
    model.zero_grad()
    score.backward()

    # Saliency map
    saliency_map = input_image.grad.data.abs().squeeze(0).cpu()
    return saliency_map


# --------------------------
# VISUALIZE SALIENCY MAPS
# --------------------------
# Take one sample from the test loader
inputs_batch, meta_batch, labels_batch = next(iter(test_loader))


for idx in range(min(4, len(inputs_batch))):  # Show up to 4 samples
    input_image = inputs_batch[idx]  # Single input
    label = labels_batch[idx].item()

    # Compute saliency map
    saliency_map = compute_saliency_map(model, input_image, label, device)

    # Plot input image and saliency map for each view
    plt.figure(figsize=(12, 4))

    # Input views
    views = ["Coronal", "Axial", "Sagittal"]
    for i, view in enumerate(views):
        plt.subplot(2, 3, i + 1)
        plt.title(f"{view} View")
        plt.imshow(input_image[i].cpu(), cmap="gray")
        plt.axis("off")

    # Saliency maps
    for i, view in enumerate(views):
        plt.subplot(2, 3, i + 4)
        plt.title(f"{view} Saliency Map")
        plt.imshow(saliency_map[i], cmap="hot")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
