# train.py
import torch
import torch.optim as optim
from model import Autoencoder  # Import the model
from data_loader import get_dataloaders  # Import data loading functions
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


# Set device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model and move it to device
model = Autoencoder().to(device)

# Define loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Get the data loaders
train_loader = get_dataloaders('NIFTI_files/')

# Training loop
epochs = 20
for epoch in range(epochs):
    running_loss = 0.0
    for data in train_loader:
        inputs = data.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

# Save the trained model
torch.save(model.state_dict(), 'autoencoder.pth')

import matplotlib.pyplot as plt






# Tracking loss across epochs
train_losses = []

# Training loop
for epoch in range(epochs):
    running_loss = 0.0
    for data in train_loader:
        inputs = data.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}")

# After training, plot the loss curve
plt.plot(train_losses)
plt.title("Training Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()




# Create a TensorBoard writer
writer = SummaryWriter('runs/autoencoder_experiment')

# In the training loop, log the loss
writer.add_scalar('Loss/train', epoch_loss, epoch)

# Log images (for example, at the end of each epoch)
writer.add_images('Original', inputs, epoch)
writer.add_images('Reconstructed', outputs, epoch)