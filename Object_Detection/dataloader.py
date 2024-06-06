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
    max_detections = 100 
    
    # print(f"Number of items in batch: {len(batch)}")
    # for item in batch:
    #     print(f"Number of boxes in item: {len(item['bboxes'])}")
        
        
    images = torch.stack([item['image_tensor'] for item in batch]) 
    padded_boxes = torch.zeros((len(batch), max_detections, 4)) 
    objectness_scores = torch.zeros((len(batch), max_detections)) 
    padded_labels = torch.full((len(batch), max_detections), 6, dtype=torch.long)  

    
    for i, item in enumerate(batch):
        num_boxes = len(item['bboxes'])
        if num_boxes > 0:
            padded_boxes[i, :num_boxes] = torch.tensor(item['bboxes'])
            objectness_scores[i, :num_boxes] = 1 
            padded_labels[i, :num_boxes] = torch.tensor(item['category'])  

    return {
        'images': images,
        'boxes': padded_boxes,
        'objectness_scores': objectness_scores,
        'labels': padded_labels,
        'img_path': [item['img_path'] for item in batch]  # List of image paths
    }

    
class SARDet(Dataset):
    def __init__(self, data_dir,imgs:list, mode:str , target_size = 512):
        self.imgs_paths = imgs
        self.target_size = target_size
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
        orig_size = img.size[0]
        img = self.transform(img)
        img = pad_resize(img , target_size=self.target_size)
        
        image_id = self.filename_to_id.get(self.imgs_paths[index])
        annotations = self.id_to_annotations.get(image_id, []) # empty list if nothing exist
        bboxes = [list((coord/orig_size)*self.target_size for coord in anno['bbox']) for anno in annotations]
        # print(f"Image {img_path} has {len(bboxes)} bounding boxes.")
        category_ids = [anno['category_id'] for anno in annotations]
        
        output = {
            "image_tensor" : img , 
            "img_path" : img_path , 
            "bboxes" : bboxes , 
            "category" : category_ids
        }
        
        return output



                
            
            
        
            
            