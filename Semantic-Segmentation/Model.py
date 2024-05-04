import torch
import torch.nn as nn
import torch.nn.functional as F


class Unet(nn.Module):
    def __init__(self):
        super(Unet , self).__init__()
        
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
        self.maxpool4 = nn.MaxPool2d(kernel_size = 2)# (512 , 28 , 28)
        #Bottleneck
        self.conv10 = nn.Conv2d(512 , 1024 , kernel_size = 3)#(1024 , 26 , 26)
        self.conv11 = nn.Conv2d(1024 , 1024 , kernel_size = 3)#(1024 , 24 , 24)
        self.up1 = nn.ConvTranspose2d(1024 , 1024 , kernel_size = 4 , padding = 1 , stride = 2) # (1024 , 56 , 56)
        self.conv12 = nn.Conv2d(1024 , 512 , kernel_size = 3) # (512 , 54 , 54)
        self.conv13 = nn.Conv2d(512 , 512 , kernel_size = 3) # (512 , 52 , 52)
        self.up2 = nn.ConvTranspose2d(512 , 512 , kernel_size = 4 , padding = 1 , stride = 2) # (512 , 104 , 104)
        self.conv14 = nn.Conv2d(512 , 256 , kernel_size = 3) # (256 , 102 , 102)
        self.conv15 = nn.Conv2d(256 , 256 , kernel_size = 3) # (256 , 100 , 100)
        self.up3 = nn.ConvTranspose2d(256 , 256 , kernel_size = 4 , padding = 1 , stride = 2) # (256 , 200 , 200)
        self.conv16 = nn.Conv2d(256 , 128 , kernel_size = 3) # (128 , 198, 198)
        self.conv17 = nn.Conv2d(128 , 128, kernel_size = 3) # (128 , 196 , 196)
        self.up4 = nn.ConvTranspose2d(128 , 128 , kernel_size = 4 , padding = 1 , stride = 2) # (128 , 392 , 392)
        self.conv18 = nn.Conv2d(128 , 64 , kernel_size=3) #(64 , 390 , 390)
        self.conv19 = nn.Conv2d(64 , 64 , kernel_size=3) #(64 , 388 , 388)
        self.conv20 = nn.Conv2d(64 , 1 , kernel_size=1) #(2 , 388 , 388)
        
        
    def forward(self , x):
            
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.maxpool3(x)
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.maxpool4(x)
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.up1(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = F.relu(self.up2(x))
        x = F.relu(self.conv14(x))
        x = F.relu(self.conv15(x))
        x = F.relu(self.up3(x))
        x = F.relu(self.conv16(x))
        x = F.relu(self.conv17(x))
        x = F.relu(self.up4(x))
        x = F.relu(self.conv18(x))
        x = F.relu(self.conv19(x))
        x = self.conv20(x)
        
        return x
        
        
        
        
        