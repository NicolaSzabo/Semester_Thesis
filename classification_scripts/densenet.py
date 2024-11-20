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


#print_config()
print(torch.version.cuda)


if torch.cuda.is_available():
    print('GPU is available. Device in use: ')
    print(torch.cuda.get_device_name(0))
else: 
    print('No GPU available. Using CPU instead.')




### Using OmegaConf to load the configuration file and automatically print the hyperparameters

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


# Function to log configuration and metrics to a YAML file
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








### Set deterministic training for reproducibility
set_determinism(seed = 0)









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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 42)

# Then, split the train again into train + val set 
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1765, random_state = 42)

print(f"Training count: {len(X_train)}, Validation count: {len(X_val)}, Test count: {len(X_test)}")



### Define the MONAI transforms for data preprocessing and augmentation
train_transform = Compose([
        LoadImage(image_only = True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        #RandRotate(range_x = np.pi / 12, prob = 0.5, keep_size = True),
        RandFlip(spatial_axis = 0, prob = 0.5),
        RandZoom(min_zoom = 0.9, max_zoom = 1.1, prob = 0.5),
    ]
)

val_transform = Compose([
        LoadImage(image_only = True),
        EnsureChannelFirst(),
        ScaleIntensity(),
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











### Define network and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DenseNet121(spatial_dims = 3, in_channels = 1, out_channels = num_class).to(device)
criterion = torch.nn.CrossEntropyLoss()
lr = config.training.lr
optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9)
epochs = config.training.epochs
val_interval = 1






# Sets up the logging for TensorBoard (helps to visualize the training process). SummaryWriter creates log files that can be opened with TensorBoard, log_dir stores the logs with unique timestamp
log_dir = f"/home/fit_member/Documents/NS_SemesterWork/Project/runs/experiment_{datetime.now().strftime('%Y-%d-%m_%H-%M')}"
# for linux: f"/home/fit_member/Documents/NS_SemesterWork/Project/runs/experiment_{datetime.now().strftime('%Y-%d-%m_%H-%M')}"
# for windows: f"file://C://Users//nicol//OneDrive//Desktop//Semester_Thesis//Project//runs//experiment_{datetime.now().strftime('%Y-%d-%m_%H-%M')}"

writer = SummaryWriter(log_dir = log_dir)




k_folds = 5


### Training loop for the model

def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, val_interval, device):
    epoch_loss_values = []
    metric_values = []

    # EPOCH LOOP
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")

        model.train()
        running_loss = 0.0 # accumulates the total loss across batches in the current epoch
        correct_train = 0 # tracks the number of correct predictions
        total_train = 0 # counts the total number of training examples seen in the epoch

        # BATCH LOOP
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, dtype = torch.float32), labels.to(device)
            optimizer.zero_grad() # ZERO GRADIENTS: clears the gradients from the previous batch
            outputs = model(inputs) # FORWARD PASS: sends the input data through the model to generate predictions
            loss = criterion(outputs, labels) # CALCULATE LOSS: computes the difference between predictions and true labels
            loss.backward() # BACKPROPAGATION: calculates gradients for each parameter in the model based on the loss
            optimizer.step() # OPTIMIZATION STEP: updates the model’s parameters based on the calculated gradients

            # Updating metrics
            running_loss += loss.item() * inputs.size(0) # ACCUMULATE LOSS: adds the batch loss (scaled by batch size) to running_loss
            _, preds = torch.max(outputs, 1) # CALCULATE PREDICTIONS: gets the predicted class labels by finding the index with the highest probability in outputs
            correct_train += torch.sum(preds == labels.data) # TRACK CORRECT PREDICTIONS: counts how many predictions in this batch match the true labels and adds to the total correct predictions for this epoch
            total_train += labels.size(0) # UPDATE TOTAL COUNT: keeps track of the total number of examples processed so far



        # Calculate and log the average loss for the epoch
        epoch_loss = running_loss / len(train_loader) # calculates the average loss for the epoch by dividing running_loss by the number of batches in train_loader
        epoch_loss_values.append(epoch_loss) # stores the average loss for this epoch, useful for tracking and plotting loss trends over time
        epoch_acc = correct_train.double() / total_train # calculates the accuracy as the ratio of correct predictions (correct_train) to the total number of samples (total_train)
        print(f"Epoch: {epoch + 1} Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}")
        
        # Log training to Tensorboard for visualization purposes
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)

        
        # if statement checks if it’s time to run a validation pass. This is controlled by val_interval, 
        # which defines how often validation should occur (e.g., every few epochs)
        if (epoch + 1) % val_interval == 0 or (epoch == epochs - 1):
            model.eval()
            # INITIALIZE VALIDATION METRICS: running_loss_val, correct_val, and total_val are reset to track loss, correct predictions, and total samples during validation
            running_loss_val = 0.0
            correct_val = 0
            total_val = 0

        # VALIDATION LOOP (No Gradient Calculation)
        with torch.no_grad(): # turns off gradient calculations to save memory and computation, which is standard practice for validation and testing
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs) # FORWARD PASS
                loss = criterion(outputs, labels) # COMPUTE LOSS
                
                running_loss_val += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data)
                total_val += labels.size(0)

                    
        val_loss = running_loss_val / len(val_loader.dataset)
        val_acc = correct_val.double() / total_val
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

        # Append validation accuracy or metric for plotting
        metric_values.append(val_acc.item())  # Store val_acc as a float

        writer.add_hparams({'lr': lr, 'batch_size': batch_size}, {'hparam/loss': val_loss})
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




def plot_metrics(epoch_loss_values, metric_values, val_interval, save_path):
    plt.figure("Train Metrics", (12, 6))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    plt.plot(range(1, len(epoch_loss_values) + 1), epoch_loss_values, label='Train Loss', color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot validation metric (e.g., accuracy)
    plt.subplot(1, 2, 2)
    plt.title("Validation Accuracy")
    if metric_values:
        plt.plot(range(1, len(metric_values) + 1), metric_values, label='Val Accuracy', color='orange')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
    else:
        print("Warning: No validation metric values to plot.")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Metrics plot saved to {save_path}")

    plt.close()






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
plot_metrics(epoch_loss_values, metric_values, val_interval, 
             save_path = f"/home/fit_member/Documents/NS_SemesterWork/Project/results/graph_{datetime.now().strftime('%Y-%d-%m_%H-%M')}")
# Windows: f"C://Users//nicol//OneDrive//Desktop//Semester_Thesis//Project//results//graph_{datetime.now().strftime('%Y-%d-%m_%H-%M')}"
# Linux: f"/home/fit_member/Documents/NS_SemesterWork/Project/results/graph_{datetime.now().strftime('%Y-%d-%m_%H-%M')}"


######## Evaluate on the test dataset
test_loss, test_acc = evaluate_model(trained_model, test_loader, device)
print(f'Final Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_acc:.4f}')


end_time = datetime.now()
end_time_str = end_time.strftime('%Y-%d-%m_%H-%M')
duration = end_time - start_time
duration_str = str(duration)


# Log results to YAML
log_results(config = config, 
            test_loss = test_loss, 
            test_acc = test_acc, 
            start_time = start_time_str, 
            end_time = end_time_str, 
            duration = duration_str, 
            model_filename = model_filename)

writer.close()