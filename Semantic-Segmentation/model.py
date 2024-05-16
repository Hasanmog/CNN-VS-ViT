import torch.nn as nn
from torch.nn import functional as F
import torch
from Attentions import Self_Attention

class Segmentor(nn.Module):
    def __init__(self , num_classes = 23):
        super().__init__()
        
        self.enc_layer1 = nn.Sequential(
            nn.Conv2d(3 , 64 , kernel_size = 3 ), # (64 , 510 , 510)
            nn.ReLU(),
            nn.BatchNorm2d(64))
        self.enc_layer2 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size = 3), # (96 , 508 , 508)
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size = 2))# (96 , 254 , 254)
        self.enc_layer3 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size = 3), # (128 , 252 , 252)
            nn.ReLU(),
            nn.BatchNorm2d(128))
        self.enc_layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 3), # (256 , 250 , 250)
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2) #   (256 , 125 , 125)         
            )
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size = 3 , stride = 2), # (256, 123 , 123)
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256 , 512 , kernel_size = 1 , stride = 1), # (512, 123 , 123)
            nn.BatchNorm2d(512),
            nn.ELU(),
            nn.Dropout(0.5),
            # Self_Attention(d_model = 512),
            nn.Conv2d(512 , 256 , kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
        )
        
        self.dec_layer1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2), # (128, 246 , 246)
            nn.Conv2d(128 , 128, kernel_size = 3), # (128, 244 , 244)
            nn.ReLU(),
            nn.BatchNorm2d(128))
        self.dec_layer2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2), # (64, 492 , 492)
            nn.Conv2d(64 , 64, kernel_size = 3), # (64, 490 , 490)
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
        # print("x1" , x_1.shape)
        # print("x2" , x_2.shape)
        # print("x3" , x_3.shape)
        # print("x4" , x_4.shape)        
        x = self.bottleneck(x_4)
        # print("bottleneck" , x.shape)
        # Upsample and concatenate with decoder output
        x_2_up = nn.Upsample(size = (x.shape[2] , x.shape[3]) , mode = 'bilinear' , align_corners=True)(x_4)
        # print("x2_up" , x_2_up.shape)
        # x = torch.cat((x, x_2_up), dim=1)
        x = self.dec_layer1(x + x_2_up)

        # Upsample and concatenate with decoder output
        x_3_up = nn.Upsample(size = (x.shape[2] , x.shape[3]) , mode = 'bilinear' , align_corners=True)(x_3)
        # x = torch.cat((x, x_3_up), dim=1)
        x = self.dec_layer2(x + x_3_up)

        # Upsample and concatenate with decoder output
        x_4_up = nn.Upsample(size = (x.shape[2] , x.shape[3]) , mode = 'bilinear' , align_corners=True)(x_1)
        # x = torch.cat((x, x_4_up), dim=1)
        x = self.dec_layer3(x + x_4_up)
        
        x = torch.clamp(x, min=-10, max=10)

        return x