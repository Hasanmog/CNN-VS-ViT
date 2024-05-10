import torch.nn as nn


class Segmentor(nn.Module):
    def __init__(self , num_classes = 23):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3 , 64 , kernel_size = 3 ), # (64 , 510 , 510)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 96, kernel_size = 3), # (96 , 508 , 508)
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size = 2), # (96 , 254 , 254)
            nn.Conv2d(96, 128, kernel_size = 3), # (128 , 252 , 252)
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size = 3), # (256 , 250 , 250)
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2) #   (256 , 125 , 125)         
            )
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size = 3 , stride = 2), # (256, 123 , 123)
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2), # (128, 246 , 246)
            nn.Conv2d(128 , 128, kernel_size = 3), # (128, 244 , 244)
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2), # (64, 492 , 492)
            nn.Conv2d(64 , 64, kernel_size = 3), # (64, 490 , 490)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64 , 1 , kernel_size=1),
            nn.Upsample(size = (512 , 512) , mode = 'bilinear' , align_corners=True),
            nn.Sigmoid()
        )
        
        
    def forward(self , x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        
        return x