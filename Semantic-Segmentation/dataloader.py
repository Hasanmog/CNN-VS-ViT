import os
import torch.nn as nn
from torch.utils.data import Dataset

class DeepGlobe_Road(Dataset):
    def __init__(self , parent_dir : str , set : str , transform):
        
        self.path = os.path.join(parent_dir , set)
        self.transform = transform
    def __len__(self):
        
        files = os.listdir(self.path)
        return int(len(files) /  2)
        
    def __getitem__(self ):
        
        data = os.listdir(self.path)
        images = []
        masks = []
        for sample in data : 
            
            if sample.endswith(".jpg"):
                images.append(sample)
                
            elif sample.endswith(".png"):
                masks.append(sample)
                
        return images , masks

        