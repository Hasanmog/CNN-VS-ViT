import json
import os
import random
import torch
import torchvision as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import  transforms
from torchvision.transforms import ToTensor



def pad_resize(img, target_size=512):
    # Get current dimensions from the image
    _, h, w = img.shape

    # Determine which dimension is smaller
    smaller_dim = min(w, h)
    larger_dim = max(w, h)
    scale_factor = target_size / larger_dim

    # Calculate new dimensions
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)

    # Create resize transform
    resize_transform = transforms.Resize((new_h, new_w))
    img = resize_transform(img)

    # Calculate padding
    # (The total padding required to reach the target size for each dimension)
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2

    # Pad to be symmetric
    padding_transform = transforms.Pad(padding=(pad_w, pad_h, target_size - new_w - pad_w, target_size - new_h - pad_h), fill=0, padding_mode='constant')
    img = padding_transform(img)

    return img

def custom_collate_fn(batch):
    # Extract components from batch
    images = [item['image_tensor'] for item in batch]
    boxes = [item['bboxes'] for item in batch]
    labels = [item['category'] for item in batch]

    # Stack images (since they are of uniform size)
    images = torch.stack(images, dim=0)

    # Boxes and labels cannot be stacked directly as their sizes vary
    return {
        'images': images,
        'boxes': boxes,  # Return as lists or convert to padded tensors if necessary
        'labels': labels
    }


class SARDet(Dataset):
    def __init__(self, data_dir, imgs:list, mode:str):
        self.imgs_paths = imgs
        
        anno_file = 'train.json' if mode in ["train", "test"] else 'val.json'
        with open(os.path.join(data_dir, anno_file), 'r') as file:
            self.anno = json.load(file)
        self.data_dir = os.path.join(data_dir , 'train') if mode in ["train", "test"] else os.path.join(data_dir , 'val')
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        self.filename_to_id = {image['file_name']: image['id'] for image in self.anno['images']}
        
        self.id_to_annotations = {}
        for anno in self.anno['annotations']:
            if anno['image_id'] in self.id_to_annotations:
                self.id_to_annotations[anno['image_id']].append({
                    'bbox': anno['bbox'],
                    'category_id': anno['category_id']
                })
            else:
                self.id_to_annotations[anno['image_id']] = [{
                    'bbox': anno['bbox'],
                    'category_id': anno['category_id']
                }]

    def __len__(self):
        return len(self.imgs_paths)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.imgs_paths[index])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = pad_resize(img)
        
        image_id = self.filename_to_id.get(self.imgs_paths[index])
        annotations = self.id_to_annotations.get(image_id, []) # empty list if nothing exist
        
        bboxes = [anno['bbox'] for anno in annotations]
        category_ids = [anno['category_id'] for anno in annotations]
        
        output = {
            "image_tensor" : img , 
            "img_path" : img_path , 
            "bboxes" : bboxes , 
            "category" : category_ids
        }
        
        return output



                
            
            
        
            
            