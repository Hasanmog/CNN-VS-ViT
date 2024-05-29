import json
import os
import random
import torch
import torchvision as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import  transforms
from torchvision.transforms import ToTensor

from torch.utils.data import Dataset
from PIL import Image
import json
import os
from torchvision import transforms

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



                
            
            
        
            
            