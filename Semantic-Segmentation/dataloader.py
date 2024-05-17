import os
import pandas as pd
import torch
import numpy as np
from torchvision import  transforms
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset 
from PIL import Image

class WHU(Dataset):
    def __init__(self, parent_dir,transform_imgs = None):
        """
        Initialize the dataset by setting the paths to images and masks,
        preparing transformations, and loading color mappings.
        
        Args:
            parent_dir (str): Path to the parent folder of the dataset.
            images (list of str): List of image file names.
            masks (list of str): List of mask file names.
        """
        self.images_dir = os.path.join(parent_dir, "Images")
        self.masks_dir = os.path.join(parent_dir, "Masks")
        self.images = os.listdir(self.images_dir)
        self.masks = os.listdir(self.masks_dir)

        if transform_imgs: #for Augmentation
            self.transform_imgs = transform_imgs
        else : 
            self.transform_imgs = transforms.Compose([
                transforms.Resize((512, 512)), 
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        self.transform_masks = transforms.Compose([
            transforms.Resize((512, 512)), 
            transforms.Grayscale(), 
            transforms.ToTensor()
        ])

        # color_mapping = pd.read_csv(os.path.join(parent_dir, "class_dict_seg.csv"))
        # self.color_map ={ (row[' r'], row[' g'], row[' b']): idx for idx, row in color_mapping.iterrows() }
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.images_dir + "/" + self.images[idx]))
        mask = Image.open(os.path.join(self.masks_dir + "/" + self.masks[idx]))

        image = self.transform_imgs(image)

        # Convert mask from RGB to class IDs based on the color mapping
        mask = self.transform_masks(mask)
        # mask = self.rgb_to_class_id(mask)

        return image, mask

    # def rgb_to_class_id(self, mask):
    #     """Convert RGB mask to class ID mask."""
    #     # Assuming mask tensor is scaled from 0 to 1, convert it back to 0-255
    #     mask_array = np.array(mask.permute(1, 2, 0) * 255, dtype=int)
    #     class_id_mask = np.zeros(mask_array.shape[:2], dtype=int)
        
    #     for rgb, class_id in self.color_map.items():
    #         # Ensure the RGB tuple is in integer form
    #         rgb = tuple(map(int, rgb))
    #         matches = (mask_array == rgb).all(axis=-1)
    #         class_id_mask[matches] = class_id

    #     return torch.tensor(class_id_mask, dtype=torch.long)


        

        
        
        
        
        