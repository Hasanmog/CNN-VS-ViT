import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectorHead(nn.Module):
    def __init__(self , hidden_dim , num_classes = 6): 
        super(DetectorHead, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        self.cls_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1),  
            nn.Conv2d(num_classes, num_classes, kernel_size=2, stride=1, padding=1)  
        )
        
        self.regression_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(), 
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=1), 
            nn.Conv2d(4, 4, kernel_size=2, stride=1, padding=1)  
        )
        
        self.centerness_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(), 
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),  
            nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=1)  
        )
        
    def forward(self , x):
        
        cls = self.cls_head(x)
        regression = self.regression_head(x)
        centerness = self.centerness_head(x)
        return cls , regression , centerness
        
class Detector(nn.Module):
    def __init__(self , num_classes = 6):
        super(Detector, self).__init__()
        
        self.encoder_1 = nn.Sequential(   #  3 x 512 x 512 
            nn.Conv2d(3 , 16 , kernel_size = 3), # 16 x 510 x 510
            nn.ReLU(),
            nn.BatchNorm2d(16), 
            )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(16 , 32 , kernel_size = 3), # 32 x 508 x 508
            nn.ReLU(),
            nn.BatchNorm2d(32),
            )
        self.encoder_3 = nn.Sequential(
            nn.Conv2d(32 , 64 , kernel_size = 3), # 64 x 506 x 506
            nn.ReLU(),
            nn.BatchNorm2d(64),
            )
        self.encoder_4 = nn.Sequential(
            nn.Conv2d(64 , 128 , kernel_size = 3 , stride = 2), # 128 x 253 x 253
            nn.ReLU(),
            nn.BatchNorm2d(128),
            )
        self.encoder_5 = nn.Sequential(
            nn.Conv2d(128 , 256 , kernel_size = 3), # 256 x 250 x 250
            nn.ReLU(),
            nn.BatchNorm2d(256),
            )
        self.encoder_6 = nn.Sequential(
            nn.Conv2d(256 , 512 , kernel_size = 4 , stride = 2), # 512 x 248 x 248
            nn.ReLU(),
            nn.BatchNorm2d(512),
            )
        
        # self.encoder_7 = nn.Sequential(
        #     nn.Conv2d(512 , 512 , kernel_size = 3 , stride = 2), # 512 x 124 x 124
        #     nn.ReLU(),
        #     nn.BatchNorm2d(512),
        #     )
        
        self.detection = DetectorHead(512 , num_classes = num_classes)
        
    def forward(self , x):
        encoders = [self.encoder_1 , self.encoder_2 , self.encoder_3 , self.encoder_4 , self.encoder_5 , self.encoder_6]
        for encoder_layer in encoders:
            x = encoder_layer(x)   
            
        cls , regression , centerness = self.detection(x) 
        
        return cls , regression , centerness