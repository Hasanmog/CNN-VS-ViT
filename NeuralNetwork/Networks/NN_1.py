import torch.nn as nn
import torch

device = 'cuda' if torch.cuda.is_available() else "cpu"
class NN_1(nn.Module):
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