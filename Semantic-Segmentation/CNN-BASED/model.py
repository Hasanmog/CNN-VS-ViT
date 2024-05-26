import torch.nn as nn
from torch.nn import functional as F
import torch

class Segmentor(nn.Module):
    def __init__(self , num_classes = 23):
        super().__init__()
        
        self.enc_layer1 = nn.Sequential(
            nn.Conv2d(3 , 64 , kernel_size = 3 ), 
            nn.ReLU(),
            nn.BatchNorm2d(64))
        self.enc_layer2 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size = 3), 
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size = 2))
        self.enc_layer3 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size = 3), 
            nn.ReLU(),
            nn.BatchNorm2d(128))
        self.enc_layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 3), 
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2)        
            )
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size = 3 , stride = 2), 
            nn.ReLU(),
        )

        
        self.dec_layer1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2), 
            nn.Conv2d(128 , 128, kernel_size = 3), 
            nn.ReLU(),
            nn.BatchNorm2d(128))
        self.dec_layer2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2),
            nn.Conv2d(64 , 64, kernel_size = 3), 
            nn.ReLU(),
            nn.BatchNorm2d(64)
            )
        self.dec_layer3 = nn.Sequential(
            nn.ConvTranspose2d(64 , 1 , kernel_size=1),
            nn.Upsample(size = (512 , 512) , mode = 'bilinear' , align_corners=True),
        )
        
    def forward(self , x):
        x_1 = self.enc_layer1(x)
        x_2 = self.enc_layer2(x_1)
        x_3 = self.enc_layer3(x_2)
        x_4 = self.enc_layer4(x_3)        
        x = self.bottleneck(x_4)

        x_2_up = nn.Upsample(size = (x.shape[2] , x.shape[3]) , mode = 'bilinear' , align_corners=True)(x_4)
        x = self.dec_layer1(x + x_2_up)

        x_3_up = nn.Upsample(size = (x.shape[2] , x.shape[3]) , mode = 'bilinear' , align_corners=True)(x_3)
        x = self.dec_layer2(x + x_3_up)

        x_4_up = nn.Upsample(size = (x.shape[2] , x.shape[3]) , mode = 'bilinear' , align_corners=True)(x_1)
        x = self.dec_layer3(x + x_4_up)
        
        x = torch.clamp(x, min=0, max=1)

        return x
