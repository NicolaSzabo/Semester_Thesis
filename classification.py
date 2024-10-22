import os
import torch
import datetime
import numpy as np
import pandas as pd
import nibabel as nib
from PIL import Image
# from keyring.util.platform_ import data_root
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from tqdm import tqdm
from PIL import Image


excel_file = 'data/PFFx_pred_data-800.xlsx'
patient_df = pd.read_excel(excel_file)
patient_df = patient_df[patient_df['images_exported'] == 'Yes']  # Filter only the rows with 'Yes'
patient_df = patient_df.dropna(subset=['OP_type', 'side_affected'])  # Drop rows where these fields are empty

patient_labels = patient_df.set_index('Pat_name')['OP_type'].to_dict()


######## Creating a Custom Dataset for our files
# Create the Dataset class using Pytorch's Dataset class: Provides structure needed to load and handle data in an efficient manner. Once this is defined in can be passed to a Dataloader to go on with
# batching, shuffling,...
# More info here: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

class HeartCTdataset(Dataset):
    def __init__(self, img_dir, patient_labels, transform = None):
        self.img_dir = img_dir
        self.patient_labels = patient_labels
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.nii.gz')] # list of image files

    def __len__(self):
        return len(self.img_files) # returns number of images in the dataset

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        patient_id = '_'.join(img_file.split('_')[:3])  # Get patient ID
        img_path = os.path.join(self.img_dir, img_file)
        
        nii_img = nib.load(img_path)
        img_data = nii_img.get_fdata()

        # Squeeze the unnecessary dimensions (from (1, 1, 2840) to (2840) for example)
        img_data = np.squeeze(img_data)
        
        # Normalize the data to the [0, 255] range and convert to uint8
        img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data)) * 255
        img_data = img_data.astype(np.uint8)
        
        # If the image is still 2D (grayscale), expand it into 3 identical channels to represent RGB. This is done because
        # the image processing library PIL expects 3 channels. The image is still grayscale, but shaped like an RGB image with
        # 3 channels
        if img_data.ndim == 2:
            img_data = np.stack([img_data] * 3, axis=-1)  

        # Conversion to PIL image:
        # PIL is a library used for opening, manipulating and saving many different image file formats. It is used here to handle image data from a NIfTI
        # format by converting it to a more familiar image format (RGB), which is commonly expected by image transformation functions in libraries like PyTorch
        img = Image.fromarray(img_data)
        
        
        if self.transform:
            img = self.transform(img)
            
            
        label = self.patient_labels.get(patient_id, 'Unknown')
            
        # Skip any samples where label is 'Unknown'
        if label == 'Unknown':
            return self.__getitem__((idx + 1) % len(self.img_files))  # Fetch the next sample
    
        label_mapping = {'Prosthesis': 0, 'Nail': 1, 'Osteosynthesis': 2}
        label = torch.tensor(label_mapping[label])
        
        return img, label # return as tuple (img, label): img is the transformed image tensor and label is the corresponding label for the patient


# Image Transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)), # Resizes all input images to a fixed size of 512x512 pixels
    transforms.RandomHorizontalFlip(), # Randomly flips the image horizontally with 50% chance (= data augmentation, helps the model to generalize better)
    # transforms.RandomRotation(15),
    # transforms.RandomResizedCrop(224),
    transforms.ToTensor(), # convert the image into a Pytorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalizes the image using the mean and standard deviation values for the ImageNet dataset.
])


# Loading the full dataset
data_dir = "data/"
dataset = XrayDataset(img_dir = data_dir, patient_labels = patient_labels, transform = transform)
len_dataset = len(dataset)


img, label = dataset[0]  

# Print the image tensor and the label
print(f"Image shape: {img.shape}")  # This prints the shape of the tensor
print(f"Label: {label}")  # This prints the label
print(len_dataset)










# Split into train, validation and test sets (80/10/10), should be played around with
train_size = int(0.8 * len_dataset)
#val_size = int(0.1 * len_dataset)
val_size = max(1, int(0.1 * len_dataset)) # If the data set is very small, so the val_size doesnt drop below 1
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


# Create data loaders for our training, validation and test datasets. The DataLoader class is used to load data in batches, making it more efficient for training and evaluation

batch_size = 32 # Each batch contrains 32 samples. TODO: maybe adjust it so the memory usage and training speed can be affected

# num_workers specifies how many subprocesses will be used to load data in parallel. When = 0, the data loading is done in the main process, which is simpler but slower. 
# Increasing the number improves speed, but consumes more system resources and can lead to issues if too many workers are used.
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 0)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = 0)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 0)

device = "cuda" if torch.cuda.is_available() else "cpu" # Checks if GPU is available, uses it if yes
print(f"Using device: {device}")

# Load a pre-trained ResNet model
model = models.efficientnet_b0(weights = True) # TODO: check if we do pretrained = False or True
# Classifier is a sequential module (including dropout layer or other intermediate layers) representing a fully connected layer. 
# Classifier[1] is the final linear classification layer.
# .in_features is the number of input features a fully connected (nn.Linear) layer expects. This value is used on the next line to initialize 
# a new linear layer to classify into the 3 classes
num_features = model.classifier[1].in_features 
model.classifier[1] = nn.Linear(num_features, 3)  # 3 classes: Nails, Osteosynthesis, Prosthesis
model = model.to(device) # Move model's parameters to device

criterion = nn.CrossEntropyLoss() # TODO: adapt the loss function? CrossEntropyLoss is generally suitable for multi-class classification
lr = 0.00001 # learning rate for the optimizer TODO: may also be adjusted. Can affect convergence and model performance
optimizer = torch.optim.Adam(model.parameters(), lr=lr) # TODO: adapt the optimizer? Alternative might be SGD

# Sets up the logging for TensorBoard (helps to visualize the training process). SummaryWriter creates log files that can be opened with TensorBoard, log_dir stores the logs with unique timestamp
log_dir = f"runs/surgical_classification/experiment_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
writer = SummaryWriter(log_dir=log_dir)




####### Training loop for the model

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs = 10):
    for epoch in range(num_epochs): # For each epoch (= complete pass through the dataset), the model is trained and validated
        print(f'Epoch {epoch + 1}/{num_epochs}')

        model.train() # Model is set to training mode. In each epoch, the training data is passed through the network, predictions are made and the loss is calculated
        running_loss = 0.0 
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward() # Calculation of gradients
            optimizer.step() # Updates the model's parameters

            # running_loss accumulates the loss, accuracy is calculated by comparing predictions (preds) with the true labels
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data)
            total_train += labels.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_train.double() / total_train
        print(f'Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}')

        # Log training to Tensorboard for visualization purposes
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)


        # Validation phase
        model.eval() # Model switches to evaluation mode. model.eval() disables gradient calculation
        running_loss_val = 0.0
        correct_val = 0
        total_val = 0

        val_images = [] # val images list to visualize in Tensorboard
        val_labels = []
        val_preds = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss_val += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data)
                total_val += labels.size(0)
                if len(val_images) < 5:  # Only collect the first 5 images
                    val_images.append(inputs.cpu()) # TODO: CPU seems to recommended, why?
                    val_labels.append(labels.cpu())
                    val_preds.append(preds.cpu())

        val_loss = running_loss_val / len(val_loader.dataset)
        val_acc = correct_val.double() / total_val
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

        writer.add_hparams({'lr': lr, 'batch_size': batch_size}, {'hparam/loss': val_loss})
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        if epoch % 5 == 0:  # Log every 5 epochs to visualize them in TensorBoard
            # Convert lists to tensors and stack them
            val_images_tensor = torch.cat(val_images)
            writer.add_images('Validation Images', val_images_tensor[:5], epoch)

    return model






######## Train the model

#num_epochs = 35 
num_epochs = 5 # TODO how many epochs?
trained_model = train_model(model = model, train_loader = train_loader,
                            val_loader = val_loader, criterion = criterion,
                            optimizer = optimizer, num_epochs = num_epochs)
torch.save(trained_model.state_dict(), "hip_fracture_model.pth") 




######### Evaluation on the test set
def evaluate_model(model, test_loader):
    """
    Evaluates a trained model on a test set dataloader and prints the accuracy.
    Parameters
    ----------
    model: torchvision.models
    test_loader : torch.utils.data.Dataloader
    """
    model.eval()
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct_test += torch.sum(preds == labels.data)
            total_test += labels.size(0)

    test_acc = correct_test.double() / total_test
    print(f'Test Accuracy: {test_acc:.4f}')

writer.close()



######### Evaluate the model on the test set
evaluate_model(trained_model, test_loader)

# The files in the runs folder are TensorBoard logs. There logs record various metrics like loss and accuracy during training and allow to visualize the training process, performance and more 
# using TensorBaord: 
# 1. Navigate to the runs directory and run this command in the terminal: tensorboard --logdir=runs
# 2. Open the provided link in a browser


# Note: validation loss >> train loss = overfitting
# Note: .105 train loss, .255, .33 val loss with resnet18 & lr=0.00001 & these tf's
# transform = transforms.Compose([
#     transforms.Resize((512, 512)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


