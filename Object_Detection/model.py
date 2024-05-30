import torch
import torch.nn as nn


class Detector(nn.Module):
    def __init__(self , num_classes = 6):
        super(Detector, self).__init__()
        
        self.encoder_1 = nn.Sequential(   # 3 x 800 x 800 / 3 x 512 x 512 / 3 x 256 x 256
            nn.Conv2d(3 , 16 , kernel_size = 3 , stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(16), 
            )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(16 , 32 , kernel_size = 3 , stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(16 , 32 , kernel_size = 3 , stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            )
        self.encoder_3 = nn.Sequential(
            nn.Conv2d(32 , 64 , kernel_size = 3 , stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
            )
        self.encoder_4 = nn.Sequential(
            nn.Conv2d(64 , 128 , kernel_size = 3 , stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            )
        self.encoder_5 = nn.Sequential(
            nn.Conv2d(128 , 256 , kernel_size = 3 , stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            )
        self.encoder_6 = nn.Sequential(
            nn.Conv2d(256 , 512 , kernel_size = 3 , stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            )
        self.encoder_7 = nn.Sequential(
            nn.Conv2d(512 , 800 , kernel_size = 3 , stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(800),
            nn.MaxPool2d(2)
            )
        self.adap_pooling = nn.AdaptiveAvgPool2d((14 , 14))
        
        self.fc = nn.Sequential(
            nn.Linear(800*14*14 , 1000),
            nn.Linear(1000 , 512) , 
            nn.Linear(512 , 256) , 
            nn.Linear(256 , 128) ,
        )
        self.classifier = nn.Linear(128 , num_classes)
        # self.detector = nn
