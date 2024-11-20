import os
import pdb
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from monai.data import decollate_batch, DataLoader
from monai.networks.nets import DenseNet121
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, Resize, ScaleIntensity, RandFlip, RandZoom
from monai.utils import set_determinism
from omegaconf import OmegaConf
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


start_time = datetime.now()
start_time_str = start_time.strftime('%Y-%d-%m_%H-%M')
model_filename = f"heart_classification_{datetime.now().strftime('%Y-%d-%m_%H-%M')}.pth"
model_save_path = f"/home/fit_member/Documents/NS_SemesterWork/Project/results/{model_filename}"
# Linux: f"/home/fit_member/Documents/NS_SemesterWork/Project/results/{model_filename}"
# Windows: f"C://Users//nicol//OneDrive//Desktop//Semester_Thesis//Project//results//{model_filename}"


# Linux: '/home/fit_member/Documents/NS_SemesterWork/Project/config.yaml'
# Windows: 'C://Users//nicol//OneDrive//Desktop//Semester_Thesis//Project//config.yaml'
config = OmegaConf.load('/home/fit_member/Documents/NS_SemesterWork/Project/config.yaml')
print(OmegaConf.to_yaml(config))


# Linux: '/home/fit_member/Documents/NS_SemesterWork/Project/results/results_log.yaml'
# Windows: 'C://Users//nicol//OneDrive//Desktop//Semester_Thesis//Project//results//results_log.yaml'
def log_results(config, test_loss, test_acc, model_filename, start_time, end_time, duration, file_path = '/home/fit_member/Documents/NS_SemesterWork/Project/results/results_log.yaml'):
    timestamp = datetime.now().strftime('%Y-%d-%m_%H-%M')
    results = {
        'model_filename': model_filename,
        'run_id': f"Run_{timestamp}",
        'start_time': start_time,
        'end_time': end_time,
        'duration': duration,
        'config': config,
        'results': {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        }
    }
    with open(file_path, 'a') as f:
        OmegaConf.save(config = OmegaConf.create(results), f = f)








### Read image filenames from the dataset folders
data_dir = config.dataset.data_dir

# This is a list comprehension: it collects items in class_names that are directories and then sorts the list
class_names = sorted(x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x)))
num_class = len(class_names)
print(f"Number of Classes: {num_class}")

# This is a nested list comprehension: Result is a list of file paths for all files in that class’s folder.
image_files = [
    [os.path.join(data_dir, class_names[i], x) for x in os.listdir(os.path.join(data_dir, class_names[i]))]
    for i in range(num_class)
]

num_each = [len(image_files[i]) for i in range(num_class)] # List of integers (only 2 values in this example), each representing the number of images in a class


# Combine all images and classes into a single list using list comprehensions
X = [file for class_files in image_files for file in class_files]  # Flatten the nested list
y = [i for i in range(num_class) for _ in range(num_each[i])]  # Repeat each class label

num_total = len(y)

# Use LoadImage from MONAI to transform NIfTI files as PyTorch tensors
first_image = LoadImage()(X[0]) # LoadImage returns (image, metadata), so [0] accesses the image
image_shape = first_image.shape


print(f"Total image count: {num_total}")
print(f"Image dimensions: {image_shape}")
print(f"Label names: {class_names}")
print(f"Label counts: {num_each}")


data = nib.load(X[0]).get_fdata()
print("Data min:", data.min(), "Data max:", data.max())
for file in X:
    data = nib.load(file).get_fdata()
    if np.isnan(data).any():
        print(f"NaNs found in {file}")







# First, split the data into a train + test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

# Then, split the train again into train + val set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.111, random_state = 42)

print(f"Training count: {len(X_train)}, Validation count: {len(X_val)}, Test count: {len(X_test)}")



### Define the MONAI transforms for data preprocessing and augmentation
train_transform = Compose([
        LoadImage(image_only = True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        Resize((256, 256, 256)),
        #RandRotate(range_x = np.pi / 12, prob = 0.5, keep_size = True),
        RandFlip(spatial_axis = 0, prob = 0.5),
        RandZoom(min_zoom = 0.9, max_zoom = 1.1, prob = 0.5),
    ]
)

val_transform = Compose([LoadImage(image_only = True),
                          EnsureChannelFirst(),
                          ScaleIntensity(),
                          Resize((256, 256, 256)),
    ]
)




class HeartClassification(Dataset):
    def __init__(self, X, y, transform = None):
        self.X = X
        self.y = y
        self.transform = transform
        self.loader = LoadImage()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.transform(self.X[index]), self.y[index]




batch_size = config.dataset.batch_size
num_workers = config.dataset.num_workers

train_dataset = HeartClassification(X_train, y_train, train_transform)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)

val_dataset = HeartClassification(X_val, y_val, val_transform)
val_loader = DataLoader(val_dataset, batch_size = batch_size, num_workers = num_workers)

test_dataset = HeartClassification(X_test, y_test, val_transform)
test_loader = DataLoader(test_dataset, batch_size = batch_size, num_workers = num_workers)










# Define a simple 3D CNN model
class Simple3DCNN(nn.Module):
    def __init__(self):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size = 3, stride = 1, padding = 1)
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size = 3, stride = 1, padding = 1)
        self.fc1 = nn.Linear(64 * 32 * 32 * 32, 128)  # Adjust based on input image size
        self.fc2 = nn.Linear(128, 2)  # Binary classification
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Initialize the model, criterion, and optimizer
lr = config.training.lr
num_epochs = config.training.num_epochs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Simple3DCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = lr, momentum = 0.9)


log_dir = f"/home/fit_member/Documents/NS_SemesterWork/Project/runs/experiment_{datetime.now().strftime('%Y-%d-%m_%H-%M')}"
# for linux: f"/home/fit_member/Documents/NS_SemesterWork/Project/runs/experiment_{datetime.now().strftime('%Y-%d-%m_%H-%M')}"
# for windows: f"file://C://Users//nicol//OneDrive//Desktop//Semester_Thesis//Project//runs//experiment_{datetime.now().strftime('%Y-%d-%m_%H-%M')}"


writer = SummaryWriter(log_dir = log_dir)


# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_loss = running_loss / total_train
        epoch_acc = correct_train / total_train
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # Log training loss and accuracy to TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)


        # Validation step
        model.eval()
        correct_val = 0
        total_val = 0
        running_val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss = running_val_loss / total_val
        val_acc = correct_val / total_val
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        # Log validation loss and accuracy to TensorBoard
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)



# Evaluation function
def evaluate_model(model, test_loader, criterion):
    model.eval()
    correct_test = 0
    total_test = 0
    running_test_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_test_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_test += (preds == labels).sum().item()
            total_test += labels.size(0)

    test_loss = running_test_loss / total_test
    test_acc = correct_test / total_test
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Log test loss and accuracy to TensorBoard
    writer.add_scalar('Loss/test', test_loss)
    writer.add_scalar('Accuracy/test', test_acc)

    return test_loss, test_acc


# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs = num_epochs)

# Run evaluation
test_loss, test_acc = evaluate_model(model, test_loader, criterion)

# Log the classification report to TensorBoard
classification_report_str = classification_report(all_labels, all_preds, target_names=['Healthy', 'Unhealthy'])
writer.add_text("Classification Report", classification_report_str)


# Close the SummaryWriter
writer.close()


end_time = datetime.now()
end_time_str = end_time.strftime('%Y-%d-%m_%H-%M')
duration = end_time - start_time
duration_str = str(duration)


# Log results to YAML
log_results(config = config, test_loss = test_loss, test_acc = test_acc, start_time = start_time_str, end_time = end_time_str, duration = duration_str, model_filename = model_filename)
