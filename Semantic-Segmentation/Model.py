import torch
import torch.nn as nn



class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3 , 3 , kernel_size = 3)#(3 , 510 , 510)
        self.conv_act = nn.ReLU()
        self.conv2 = nn.Conv2d(3 , 64 , kernel_size = 3) # (64 , 508 , 508 )
        self.conv3 = nn.Conv2d(64 , 64 , kernel_size = 3) # (64 , 506 , 506)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2) # (64 , 253 , 253)
        self.conv4 = nn.Conv2d(64 , 128 , kernel_size = 3)# (128 , 251 , 251)
        self.conv5 = nn.Conv2d(128 , 128 , kernel_size = 4)# (128 , 248 , 248)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)# (128 , 124 , 124)
        self.conv6 = nn.Conv2d(128 , 256 , kernel_size = 3)#(256 , 122 , 122)
        self.conv7 = nn.Conv2d(256 , 256 , kernel_size = 3)#(256 , 120 , 120)
        self.maxpool3 = nn.MaxPool2d(kernel_size = 2)# (256 , 60 , 60)
        self.conv8 = nn.Conv2d(256 , 512 , kernel_size = 3)#(512 , 58 , 58)
        self.conv9 = nn.Conv2d(512 , 512 , kernel_size = 3)#(512 , 56 , 56)
        self.maxpool3 = nn.MaxPool2d(kernel_size = 2)# (512 , 28 , 28)
        self.conv10 = nn.Conv2d(512 , 1024 , kernel_size = 3)#(1024 , 26 , 26)
        self.conv11 = nn.Conv2d(1024 , 1024 , kernel_size = 3)#(1024 , 24 , 24)
        
        
        
        