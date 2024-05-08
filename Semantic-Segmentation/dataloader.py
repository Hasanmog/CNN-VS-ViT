import os
import pandas as pd
import torch
import numpy as np
from torchvision import  transforms
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset 
from PIL import Image

class Drone(Dataset):
    def __init__(self, parent_dir, images, masks):
        """
        Initialize the dataset by setting the paths to images and masks,
        preparing transformations, and loading color mappings.
        
        Args:
            parent_dir (str): Path to the parent folder of the dataset.
            images (list of str): List of image file names.
            masks (list of str): List of mask file names.
        """
        images_dir = os.path.join(parent_dir, "dataset/semantic_drone_dataset/original_images")
        rgb_masks_dir = os.path.join(parent_dir, "RGB_color_image_masks/RGB_color_image_masks")
        self.images_paths = [os.path.join(images_dir, image) for image in images]
        self.masks_paths = [os.path.join(rgb_masks_dir, mask) for mask in masks]

        self.transform_imgs = transforms.Compose([
            transforms.Resize((512, 512)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transform_masks = transforms.Compose([
            transforms.Resize((512, 512)),  
            transforms.ToTensor()
        ])

        color_mapping = pd.read_csv(os.path.join(parent_dir, "class_dict_seg.csv"))
        self.color_map ={ (row[' r'], row[' g'], row[' b']): idx for idx, row in color_mapping.iterrows() }
        
    def __len__(self):
        return len(self.images_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.images_paths[idx]).convert('RGB')
        mask = Image.open(self.masks_paths[idx]).convert('RGB')

        image = self.transform_imgs(image)

        # Convert mask from RGB to class IDs based on the color mapping
        mask = self.transform_masks(mask)
        mask = self.rgb_to_class_id(mask)

        return image, mask

    def rgb_to_class_id(self, mask):
        """Convert RGB mask to class ID mask."""
        mask_array = np.array(mask)
        class_id_mask = np.zeros(mask_array.shape[:2], dtype=int)
        
        for rgb, class_id in self.color_map.items():
            matches = (mask_array == rgb).all(axis=-1)
            class_id_mask[matches] = class_id
        
        return torch.tensor(class_id_mask, dtype=torch.long)
        

        
        
        
        
        