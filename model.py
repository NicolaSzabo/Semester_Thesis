import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder: Conv layers followed by Transformer blocks
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding = 1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Add transformer layer or blocks here
        )
        
        # Decoder: Reconstruct the image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride = 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder()
