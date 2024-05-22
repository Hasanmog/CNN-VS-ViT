import torch.nn as nn
import torch
import loralib as lora

device = 'cuda' if torch.cuda.is_available() else "cpu"
class NN_1(nn.Module):
    # Classifier #1
    def __init__(self , num_classes = 10 , device = device ): # batch --> (BATCH_SIZE , 3 , 64 , 64 )
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels = 3 , out_channels= 16 , kernel_size=3) # (16 , 62 , 62)
        self.batch_norm_1 = nn.BatchNorm2d(16)
        self.act_1 = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(in_channels = 16 ,out_channels = 64 , kernel_size = 5) # (64 , 58 , 58)
        self.batch_norm_2  = nn.BatchNorm2d(64)
        self.act_2 = nn.ReLU()
        self.flatten = nn.Flatten() # (64 , 58 ,58)
        self.fc1 = nn.Linear(64*58*58 , 16)
        self.fc2 = nn.Linear(16 , num_classes)
        self.act_3 = nn.LogSoftmax() ## nn.logsoftmax
        
    def forward(self, x):
        x = x.to(device) 
        conv2d_1 = self.conv2d_1(x)
        batch_norm1 = self.batch_norm_1(conv2d_1)
        act_1 = self.act_1(batch_norm1)
        conv2d_2 = self.conv2d_2(act_1)
        batch_norm_2 = self.batch_norm_2(conv2d_2)
        act_2 = self.act_2(batch_norm_2)
        flatten = self.flatten(act_2)
        fc1 = self.fc1(flatten)
        fc2 = self.fc2(fc1)
        logits = self.act_3(fc2)
        return logits
    
    
class NN_2(nn.Module):
    # Classifier #3
    def __init__(self , num_classes = 10 , device = device ): # batch --> (BATCH_SIZE , 3 , 64 , 64 )
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels = 3 , out_channels= 16 , kernel_size=3) # (16 , 62 , 62)
        self.batch_norm_1 = nn.BatchNorm2d(16)
        self.act_1 = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(in_channels = 16 , out_channels= 32 , kernel_size=3) # (16 , 60 , 60)
        self.batch_norm_2 = nn.BatchNorm2d(32)
        self.act_2 = nn.ReLU()
        self.conv2d_3 = nn.Conv2d(in_channels = 32 ,out_channels = 64 , kernel_size = 5) # (64 , 56 , 56)
        self.batch_norm_3  = nn.BatchNorm2d(64)
        self.act_3 = nn.ReLU()
        self.flatten = nn.Flatten() # (64 , 56 ,56)
        self.fc1 = nn.Linear(64*56*56 , 128)
        self.fc2 = nn.Linear(128 , 64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32 , num_classes)
        self.act_3 = nn.LogSoftmax(dim = 1) ## nn.logsoftmax
        
    def forward(self, x):
        x = x.to(device) 
        batch_norm1 = self.batch_norm_1(self.conv2d_1(x))
        act_1 = self.act_1(batch_norm1)
        batch_norm_2 = self.batch_norm_2(self.conv2d_2(act_1))
        act_2 = self.act_2(batch_norm_2)
        batch_norm_3 = self.batch_norm_3(self.conv2d_3(act_2))
        act_3 = self.act_3(batch_norm_3)
        fc1 = self.fc1(self.flatten(act_3))
        fc = self.fc4(self.fc3(self.fc2(fc1)))
        logits = self.act_3(fc)
        return logits
    
class NN_3(nn.Module):
    # Classifier #2
    def __init__(self , num_classes = 10 , device = device ): # batch --> (BATCH_SIZE , 3 , 64 , 64 )
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels = 3 , out_channels= 16 , kernel_size=3) # (16 , 62 , 62)
        self.maxpool2d = nn.MaxPool2d(kernel_size = 2) # (16 , 31 , 31)
        self.batch_norm_1 = nn.BatchNorm2d(16)
        self.act_1 = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(in_channels = 16 , out_channels= 32 , kernel_size=3) # (16 , 29 , 29)
        self.batch_norm_2 = nn.BatchNorm2d(32)
        self.act_2 = nn.ReLU()
        self.conv2d_3 = nn.Conv2d(in_channels = 32 ,out_channels = 64 , kernel_size = 3) # (64 , 27 , 27)
        self.batch_norm_3  = nn.BatchNorm2d(64)
        self.act_3 = nn.ReLU()
        self.conv2d_4 = nn.Conv2d(in_channels = 64 ,out_channels = 128 , kernel_size = 5) # (128 , 23 ,23)
        self.batch_norm_4  = nn.BatchNorm2d(128)
        self.act_4 = nn.ReLU()
        self.flatten = nn.Flatten() # (64 , 56 ,56)
        self.fc1 = nn.Linear(128*23*23 , 128)
        self.fc2 = nn.Linear(128 , 64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32 , num_classes)
        self.act_5 = nn.LogSoftmax() ## nn.logsoftmax
        
    def forward(self, x):
        x = x.to(device) 
        batch_norm1 = self.batch_norm_1(self.maxpool2d(self.conv2d_1(x)))
        act_1 = self.act_1(batch_norm1)
        batch_norm_2 = self.batch_norm_2(self.conv2d_2(act_1))
        act_2 = self.act_2(batch_norm_2)
        batch_norm_3 = self.batch_norm_3(self.conv2d_3(act_2))
        act_3 = self.act_3(batch_norm_3)
        batch_norm_4 = self.batch_norm_4(self.conv2d_4(act_3))
        act_4 = self.act_4(batch_norm_4)
        fc1 = self.fc1(self.flatten(act_4))
        fc = self.fc4(self.fc3(self.fc2(fc1)))
        logits = self.act_5(fc)
        return logits
    
    
class NN_4(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2)
        self.batch_norm_1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        
        self.conv2d_2 = nn.Conv2d(16, 32, 3)
        self.batch_norm_2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        
        self.conv2d_3 = nn.Conv2d(32, 64, 3)
        self.batch_norm_3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        
        self.conv2d_4 = nn.Conv2d(64, 128, 2)
        self.batch_norm_4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        
        self.conv2d_5 = nn.Conv2d(128, 256, 3)
        self.batch_norm_5 = nn.BatchNorm2d(256) # (256 , 120 ,120)
        self.relu5 = nn.ReLU()
        
        self.conv2d_6 = nn.Conv2d(256, 256, 4) #(256 , 117, 117)
        self.maxpool2d_6 = nn.MaxPool2d(3) # (256 , 39 , 39) 
        self.batch_norm_6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256*19*19, 128)
        self.fc2 = nn.Linear(128, 64)
        self.drop = nn.Dropout(0.4)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.relu1(self.batch_norm_1(self.maxpool2d(self.conv2d_1(x))))
        x = self.relu2(self.batch_norm_2(self.conv2d_2(x)))
        x = self.relu3(self.batch_norm_3(self.conv2d_3(x)))
        x = self.relu4(self.batch_norm_4(self.conv2d_4(x)))
        x = self.relu5(self.batch_norm_5(self.conv2d_5(x)))
        x = self.relu6(self.batch_norm_6(self.maxpool2d_6(self.conv2d_6(x))))
        x = self.flatten(self.pool(x))
        x = self.fc1(x)
        x = self.drop(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        return x  # Return logits directly if using CrossEntropyLoss
