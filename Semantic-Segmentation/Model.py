import torch
import torch.nn as nn
import torch.nn.functional as F


class Unet(nn.Module):
    def __init__(self):
        super(Unet , self).__init__()
        #encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, padding=1, stride=2) , # Reduce output channels and kernel size,
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, padding=1, stride=2),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, padding=1, stride=2),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, padding=1, stride=2),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
        
        
        
        
        