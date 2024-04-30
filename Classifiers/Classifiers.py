import torch.nn as nn
import torch

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
        self.act_3 = nn.LogSoftmax() ## nn.logsoftmax
        
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
        self.act_3 = nn.LogSoftmax() ## nn.logsoftmax
        
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
        logits = self.act_3(fc)
        return logits
    
    
class NN_4(nn.Module):
        # Classifier #4
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
            self.conv2d_4 = nn.Conv2d(in_channels = 64 ,out_channels = 128 , kernel_size = 2) # (128 , 26 ,26)
            self.batch_norm_4  = nn.BatchNorm2d(128)
            self.act_4 = nn.ReLU()
            self.conv2d_5 = nn.Conv2d(in_channels = 128 ,out_channels = 256 , kernel_size = 3) # (128 , 24 ,24)
            self.maxpool2d_2 = nn.MaxPool2d(kernel_size = 3 ) # (128 , 8 , 8)
            self.batch_norm_5  = nn.BatchNorm2d(256)
            self.act_5 = nn.ReLU()
            self.flatten = nn.Flatten() # (64 , 56 ,56)
            self.fc1 = nn.Linear(256*8*8 , 64)
            self.fc2 = nn.Linear(64 , 32)
            self.fc3 = nn.Linear(32,16)
            self.fc4 = nn.Linear(16 , num_classes)
            self.act_3 = nn.LogSoftmax() ## nn.logsoftmax
            
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
            batch_norm_5 = self.batch_norm_5(self.maxpool2d_2(self.conv2d_5(act_4)))
            act_5 = self.act_5(batch_norm_5)
            fc1 = self.fc1(self.flatten(act_5))
            fc = self.fc4(self.fc3(self.fc2(fc1)))
            logits = self.act_3(fc)
            return logits