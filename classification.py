import os
import gc
import shutil
import tempfile
import matplotlib.pyplot as plt
import nibabel as nib
import PIL
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
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
    Resize
)
from monai.utils import set_determinism
from omegaconf import OmegaConf
from datetime import datetime


#print_config()
print(torch.version.cuda)


if torch.cuda.is_available():
    print('GPU is available. Device in use: ')
    print(torch.cuda.get_device_name(0))
else: 
    print('No GPU available. Using CPU instead.')

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#print(os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))

#torch.cuda.empty_cache()
#gc.collect()

#os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"  # Disables OpenCV GUI dependencies









### Using OmegaConf to load the configuration file and automatically print the hyperparameters

start_time = datetime.now()
start_time_str = start_time.strftime('%Y-%d-%m_%H-%M')
model_filename = f"heart_classification_{datetime.now().strftime('%Y-%d-%m_%H-%M')}.pth"
model_save_path = rf"C:\Users\nicol\OneDrive\Desktop\Semester_Thesis\Project\results\{model_filename}"
# Linux: f"'/home/fit_member/Documents/NS_SemesterWork/{model_filename}"
# Windows: rf"'C:\Users\nicol\OneDrive\Desktop\Semester_Thesis\Project\{model_filename}"


# Linux: '/home/fit_member/Documents/NS_SemesterWork/config.yaml'
# Windows: 'C:\Users\nicol\OneDrive\Desktop\Semester_Thesis\Project\config.yaml'
config = OmegaConf.load(r'C:\Users\nicol\OneDrive\Desktop\Semester_Thesis\Project\config.yaml')
print(OmegaConf.to_yaml(config))

# Function to log configuration and metrics to a YAML file
# Linux: '/home/fit_member/Documents/NS_SemesterWork/results_log.yaml'
# Windows: 'C:\Users\nicol\OneDrive\Desktop\Semester_Thesis\Project\results_log.yaml'
def log_results(config, test_loss, test_acc, model_filename, start_time, end_time, duration, file_path = r'C:\Users\nicol\OneDrive\Desktop\Semester_Thesis\Project\results\results_log.yaml'):
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








### Set deterministic training for reproducibility
set_determinism(seed = 0)









### Read image filenames from the dataset folders


data_dir = config.dataset.data_dir

class_names = sorted(x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x)))
num_class = len(class_names)
print(f"Number of classes: {num_class}")

image_files = [
    [os.path.join(data_dir, class_names[i], x) for x in os.listdir(os.path.join(data_dir, class_names[i]))]
    for i in range(num_class)
]

num_each = [len(image_files[i]) for i in range(num_class)] # List of integers, each representing the number of images in a class

# Combine all images and classes into a single listJust
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


data = nib.load(image_files_list[0]).get_fdata()
print("Data min:", data.min(), "Data max:", data.max())
for file in image_files_list:
    data = nib.load(file).get_fdata()
    if np.isnan(data).any():
        print(f"NaNs found in {file}")




### Randomly pick images from the dataset to visualize and check
"""
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
"""






### Prepare training, validation and test data lists
val_frac = 0.1
test_frac = 0.1
length = len(image_files_list)
indices = np.arange(length)
np.random.shuffle(indices)

test_split = int(length * test_frac)
val_split = int(val_frac * length) + test_split
test_indices = indices[:test_split]
val_indices = indices[test_split:val_split]
train_indices = indices[val_split:]

train_x = [image_files_list[i] for i in train_indices]
train_y = [image_class[i] for i in train_indices]
val_x = [image_files_list[i] for i in val_indices]
val_y = [image_class[i] for i in val_indices]
test_x = [image_files_list[i] for i in test_indices]
test_y = [image_class[i] for i in test_indices]

print(f"Training count: {len(train_x)}, Validation count: {len(val_x)}, Test count: {len(test_x)}")









### Define the MONAI transforms for data preprocessing and augmentation
train_transforms = Compose(
    [
        LoadImage(image_only = True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        Resize((128, 128, 128)),
        RandRotate(range_x = np.pi / 12, prob = 0.5, keep_size = True),
        RandFlip(spatial_axis = 0, prob = 0.5),
        RandZoom(min_zoom = 0.9, max_zoom = 1.1, prob = 0.5),
    ]
)

val_transforms = Compose([LoadImage(image_only = True),
                          EnsureChannelFirst(),
                          ScaleIntensity(),
                          Resize((128, 128, 128)),
    ]
)

y_pred_trans = Compose([Activations(softmax = True)])
y_trans = Compose(AsDiscrete(to_onehot = num_class))


class HeartClassification(Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]


batch_size = config.dataset.batch_size
num_workers = config.dataset.num_workers

train_ds = HeartClassification(train_x, train_y, train_transforms)
train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True, num_workers = num_workers)

val_ds = HeartClassification(val_x, val_y, val_transforms)
val_loader = DataLoader(val_ds, batch_size = batch_size, num_workers = num_workers)

test_ds = HeartClassification(test_x, test_y, val_transforms)
test_loader = DataLoader(test_ds, batch_size = batch_size, num_workers = num_workers)









### Define network and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DenseNet121(spatial_dims = 3, in_channels = 1, out_channels = num_class).to(device)
criterion = torch.nn.CrossEntropyLoss()
learning_rate = config.training.learning_rate
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
epochs = config.training.epochs
val_interval = 1






# Sets up the logging for TensorBoard (helps to visualize the training process). SummaryWriter creates log files that can be opened with TensorBoard, log_dir stores the logs with unique timestamp
log_dir = rf"C:\Users\nicol\OneDrive\Desktop\Semester_Thesis\Project\runs\{datetime.now().strftime('%Y-%d-%m_%H-%M')}"
# for linux: f"'/home/fit_member/Documents/NS_SemesterWork/_{datetime.now().strftime('%Y-%d-%m_%H-%M')}"
# for windows: rf"'C:\Users\nicol\OneDrive\Desktop\Semester_Thesis\Project\_{datetime.now().strftime('%Y-%d-%m_%H-%M')}"

writer = SummaryWriter(log_dir = log_dir)







### Training loop for the model

def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, val_interval, device):
    epoch_loss_values = []
    metric_values = []

    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")

        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, dtype = torch.float32), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data)
            total_train += labels.size(0)



        # Calculate and log the average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_loss_values.append(epoch_loss)
        epoch_acc = correct_train.double() / total_train
        print(f"Epoch: {epoch + 1} Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}")
        
        # Log training to Tensorboard for visualization purposes
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)

        
        # Run validation only at the specified intervals
        if (epoch + 1) % val_interval == 0 or (epoch == epochs - 1):
            model.eval()
            running_loss_val = 0.0
            correct_val = 0
            total_val = 0
       
        model.eval()
        running_loss_val = 0.0
        correct_val = 0
        total_val = 0
        


        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss_val += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data)
                total_val += labels.size(0)

                    
        val_loss = running_loss_val / len(val_loader.dataset)
        val_acc = correct_val.double() / total_val
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

        writer.add_hparams({'lr': learning_rate, 'batch_size': batch_size}, {'hparam/loss': val_loss})
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

    return model, epoch_loss_values, metric_values







def evaluate_model(model, test_loader, device):
    model.eval()
    correct_test = 0.0
    total_test = 0.0
    running_loss_test = 0.0
    criterion = torch.nn.CrossEntropyLoss()


    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss_test += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_test += torch.sum(preds == labels.data)
            total_test += labels.size(0)

    test_loss = running_loss_test / len(test_loader.dataset)
    test_acc = correct_test.double() / total_test

    # Log the test accuracy and loss to TensorBoard
    writer.add_scalar('Loss/test', test_loss)
    writer.add_scalar('Accuracy/test', test_acc)

        
    return test_loss, test_acc








def plot_metrics(epoch_loss_values, metric_values, val_interval):
    plt.figure("Train Metrics", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    plt.plot(range(1, len(epoch_loss_values) + 1), epoch_loss_values, label = 'Train Loss')
    plt.xlabel("Epoch")
   

    plt.subplot(1, 2, 2)
    plt.title("Validation AUC")
    plt.plot([val_interval * (i + 1) for i in range(len(metric_values))], metric_values, label = 'Val AUC')
    plt.xlabel("Epoch")

    plt.legend()
    plt.show()





########### Train the model
trained_model, epoch_loss_values, metric_values = train_model(
    model = model,
    train_loader = train_loader,
    val_loader = val_loader,
    optimizer = optimizer,
    criterion = criterion,
    epochs = epochs,
    val_interval = val_interval,
    device = device,
)
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")



######## Plot training metrics
plot_metrics(epoch_loss_values, metric_values, val_interval)



######## Evaluate on the test dataset
test_loss, test_acc = evaluate_model(trained_model, test_loader, device)
print(f'Final Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_acc:.4f}')


end_time = datetime.now()
end_time_str = end_time.strftime('%Y-%d-%m_%H-%M')
duration = end_time - start_time
duration_str = str(duration)


# Log results to YAML
log_results(config = config, test_loss = test_loss, test_acc = test_acc, start_time = start_time_str, end_time = end_time_str, duration = duration_str, model_filename = model_filename)

writer.close()