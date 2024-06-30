import torch
import torch.nn as nn
import torch.nn.functional as F


class Regression_Head(nn.Module):
    def __init__(self , hidden_dim , num_classes = 6): 
        super(Regression_Head, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        self.regression_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(), 
            nn.BatchNorm2d(128) , 
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.BatchNorm2d(64) ,
            nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=1), 
            nn.ReLU() , 
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 4, kernel_size=2, stride=1, padding=1)  
        )
        
    def forward(self , x):
        
        regression = self.regression_head(x)
        return  regression 
    
class Classification_Head(nn.Module):
    def __init__(self , hidden_dim , num_classes = 6): 
        super(Classification_Head, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        self.cls_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.BatchNorm2d(128) , 
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.BatchNorm2d(64) , 
            nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1),  
            nn.ReLU() , 
            nn.BatchNorm2d(num_classes),
            nn.Conv2d(num_classes, num_classes, kernel_size=2, stride=1, padding=1)  
        )
        
    def forward(self , x):
        
        classification = self.cls_head(x)
        return  classification
    
class Centerness_Head(nn.Module):
    def __init__(self , hidden_dim , num_classes = 6): 
        super(Centerness_Head, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        self.center_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(), 
            nn.BatchNorm2d(128) , 
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.BatchNorm2d(64) , 
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),  
            nn.ReLU() , 
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=1)  
        ) 
           
        
    def forward(self , x):
        
        centerness = self.center_head(x)
        return  centerness
         
class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 16 x 512 x 512
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 32 x 512 x 512
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.encoder_3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64 x 512 x 512
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.encoder_4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128 x 256 x 256
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.encoder_5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 256 x 256 x 256
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        self.encoder_6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 512 x 128 x 128
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, 256, kernel_size=1)
            for in_channels in [16, 32, 64, 128, 256, 512]
        ])
        
    def forward(self, x):
        encoders = [self.encoder_1, self.encoder_2, self.encoder_3, self.encoder_4, self.encoder_5, self.encoder_6]
        encoder_outputs = []
        
        for encoder in encoders:
            x = encoder(x)
            encoder_outputs.append(x)
        
        encoder_outputs = [lateral_conv(feature_map) for lateral_conv, feature_map in zip(self.lateral_convs, encoder_outputs)]
        
        p6 = encoder_outputs[-1]
        p5 = F.interpolate(p6, size=encoder_outputs[-2].shape[2:], mode='nearest') + encoder_outputs[-2]
        p4 = F.interpolate(p5, size=encoder_outputs[-3].shape[2:], mode='nearest') + encoder_outputs[-3]
        p3 = F.interpolate(p4, size=encoder_outputs[-4].shape[2:], mode='nearest') + encoder_outputs[-4]
        p2 = F.interpolate(p3, size=encoder_outputs[-5].shape[2:], mode='nearest') + encoder_outputs[-5]
        p1 = F.interpolate(p2, size=encoder_outputs[-6].shape[2:], mode='nearest') + encoder_outputs[-6]
        
        return [p1, p2, p3, p4, p5, p6]
            
            
class Detector(nn.Module):
    def __init__(self , num_classes = 6):
        super(Detector, self).__init__()
        self.FPN = FPN()
        self.regression_head = Regression_Head(hidden_dim = 256)
        self.centerness_head = Centerness_Head(hidden_dim=256)
        self.class_head = Classification_Head(hidden_dim=256)
        self.num_classes = num_classes
        
    def forward(self, x):
        featuremaps = self.FPN(x)  # Get the list of Ps from FPN
        
        cls_outputs = []
        regression_outputs = []
        centerness_outputs = []
        
        for p in featuremaps:
            cls_output = self.class_head(p)
            regression_output = self.regression_head(p)
            centerness_output = self.centerness_head(p)
            
            cls_outputs.append(cls_output.permute(0, 2, 3, 1).reshape(-1, self.num_classes))
            regression_outputs.append(regression_output.permute(0, 2, 3, 1).reshape(-1, 4))
            centerness_outputs.append(centerness_output.permute(0, 2, 3, 1).reshape(-1))
        
        # Concatenate all outputs
        cls_outputs = torch.cat(cls_outputs, dim=0)
        regression_outputs = torch.cat(regression_outputs, dim=0)
        centerness_outputs = torch.cat(centerness_outputs, dim=0)
        
        return cls_outputs, regression_outputs, centerness_outputs
        
        
    