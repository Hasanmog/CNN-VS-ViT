import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset


class EuroSAT(Dataset):
    def __init__(self, parent_dir , data, transform):
        
        self.image_paths = [item[0] for item in data]
        self.labels = [item[1] for item in data]
        self.parent_dir = parent_dir
        self.transform = transform 
        
    def __len__(self ):
        
        return len(self.labels)
    
    def __getitem__(self , idx):
        
        image_path = os.path.join(self.parent_dir , self.image_paths[idx])
        image = Image.open(image_path)
        
        if self.transform:
            img = self.transform(image)
        
        label = self.labels[idx]
         
        return img , label        
        
        
    
        
    
    
        
    