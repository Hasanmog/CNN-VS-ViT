import os
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

class WHU_bldg(Dataset):
    def __init__(self , parent_dir : str , set : str , transform_rgb = None , transform_grey = None):
        
        self.images_dir = os.path.join(parent_dir , set , 'Images')
        self.masks_dir = os.path.join(parent_dir , set , 'Masks')
        self.transform_rgb = transform_rgb
        self.transform_grey = transform_grey
        
        self.images = [os.path.join(self.images_dir, f) for f in sorted(os.listdir(self.images_dir))]
        self.masks = [os.path.join(self.masks_dir , f) for f in sorted(os.listdir(self.masks_dir))]
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self , idx ):
        
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")
        
        if self.transform_rgb:
            image = self.transform_rgb(image)
        if self.transform_grey:
            mask = self.transform_grey(mask)
        
        return image , mask

        