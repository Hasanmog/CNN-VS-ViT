import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import nms

class DetectorHead(nn.Module):
    def __init__(self ,in_channels ,  anchors , num_classes = 6 ,):
        super(DetectorHead, self).__init__()
        
        self.pred = nn.Conv2d(in_channels , anchors * (1+4+num_classes) , kernel_size = 3 , padding = 1)
    def forward(self , x):
        
        x = self.pred(x)
        return x
        
class Detector(nn.Module):
    def __init__(self , num_classes = 6):
        super(Detector, self).__init__()
        
        self.encoder_1 = nn.Sequential(   # 3 x 800 x 800 / 3 x 512 x 512 / 3 x 256 x 256
            nn.Conv2d(3 , 16 , kernel_size = 3),
            nn.ReLU(),
            nn.BatchNorm2d(16), 
            )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(16 , 32 , kernel_size = 3 , stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            )
        self.encoder_3 = nn.Sequential(
            nn.Conv2d(32 , 64 , kernel_size = 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            )
        self.encoder_4 = nn.Sequential(
            nn.Conv2d(64 , 128 , kernel_size = 3 , stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            )
        self.encoder_5 = nn.Sequential(
            nn.Conv2d(128 , 256 , kernel_size = 3),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            )
        self.encoder_6 = nn.Sequential(
            nn.Conv2d(256 , 512 , kernel_size = 3 , stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            )
        
        self.detection = DetectorHead(512 , anchors = 5 , num_classes = num_classes)
        
    def forward(self , x):
        encoders = [self.encoder_1 , self.encoder_2 , self.encoder_3 , self.encoder_4 , self.encoder_5 , self.encoder_6]
        for encoder_layer in encoders:
            x = encoder_layer(x)    
        x = self.detection(x)
        return x

def decode_outputs(outputs, score_threshold=0.5, iou_threshold=0.5, num_classes=6, anchors=5):
    
    '''
    outputs: Tensor of size (B , anchor*(4+1+num_classes) , H , W)
    x 
    y
    h
    w
    object score
    c0
    c1
    c2
    c3
    c4
    c5
    '''

    B, C, H, W = outputs.shape  # Outputs shape (B, 55, H, W)
    
    outputs = outputs.view(B, anchors, 5 + num_classes, H, W)
    outputs = outputs.permute(0, 1, 3, 4, 2).contiguous()

    boxes = outputs[..., :4]
    scores = torch.sigmoid(outputs[..., 4])
    class_probs = torch.sigmoid(outputs[..., 5:])

    return boxes , scores,  class_probs