import json
import os
import numpy as np
import torch
import math
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def normalize_bbox(bbox, orig_size, target_size, stride):
    x0, y0, w, h = bbox
    x1 = x0 + w
    y1 = y0 + h

    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    # Calculate grid coordinates
    grid_x0 = int(x0 / orig_size * (target_size // stride))
    grid_y0 = int(y0 / orig_size * (target_size // stride))
    grid_x1 = int(x1 / orig_size * (target_size // stride))
    grid_y1 = int(y1 / orig_size * (target_size // stride))

    # Clamp grid coordinates to be within valid range
    grid_x0 = max(0, min(grid_x0, target_size // stride - 1))
    grid_y0 = max(0, min(grid_y0, target_size // stride - 1))
    grid_x1 = max(0, min(grid_x1, target_size // stride - 1))
    grid_y1 = max(0, min(grid_y1, target_size // stride - 1))

    grid_cx = int(cx / orig_size * (target_size // stride))
    grid_cy = int(cy / orig_size * (target_size // stride))

    grid_cx = max(0, min(grid_cx, target_size // stride - 1))
    grid_cy = max(0, min(grid_cy, target_size // stride - 1))

    return grid_x0, grid_y0, grid_x1, grid_y1, grid_cx, grid_cy

def distanceaway(x, y, x0, y0, x1, y1):
    '''
    Function to calculate the distance from the center of a grid (x, y) to a bounding box of normalized coordinates (x0,y0,x1,y1).
    
    Returns: 
    l : left distance
    t : top distance
    r : right distance
    b : bottom distance
    '''
    # Calculate the center of the current grid cell
    # print(f"x:{x} , y:{y} , x0:{x0} , yo:{y0},x1:{x1} , y1:{y1} ")
    cx = x + 0.5
    cy = y + 0.5
    # print(f"cx:{cx} , cy:{cy}")
    l = cx - x0
    t = cy - y0
    r = x1 - cx
    b = y1 - cy
    return l, t, r, b



def adjust_bbox_with_padding(bbox, pad_w, pad_h):
    x0, y0, w, h = bbox
    x0 += pad_w
    y0 += pad_h
    return [x0, y0, w, h]

def resize_and_pad_image(img, target_size=512):
    _, h, w = img.shape

    smaller_dim = min(w, h)
    larger_dim = max(w, h)
    scale_factor = target_size / larger_dim

    new_w, new_h = int(w * scale_factor), int(h * scale_factor)

    resize_transform = transforms.Resize((new_h, new_w))
    img = resize_transform(img)

    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2

    padding_transform = transforms.Pad(padding=(pad_w, pad_h, target_size - new_w - pad_w, target_size - new_h - pad_h), fill=0, padding_mode='constant')
    img = padding_transform(img)

    return img, pad_w, pad_h, new_w, new_h

def calculate_centerness(l, t, r, b):
    epsilon = 1e-6  # small value to avoid division by zero
    min_lr = max(min(l, r), epsilon)
    max_lr = max(max(l, r), epsilon)
    min_tb = max(min(t, b), epsilon)
    max_tb = max(max(t, b), epsilon)
    centerness = math.sqrt(
        (min_lr / max_lr) * 
        (min_tb / max_tb)
    )
    return centerness

class SARDet(Dataset):
    def __init__(self, data_dir, imgs, mode, target_size=512, stride=16, num_classes=6):
        self.imgs_paths = imgs
        self.target_size = target_size
        self.num_classes = num_classes
        self.stride = stride

        anno_file = 'train.json' if mode in ["train", "test"] else 'val.json'
        with open(os.path.join(data_dir, anno_file), 'r') as file:
            self.anno = json.load(file)
        self.data_dir = os.path.join(data_dir, 'train') if mode in ["train", "test"] else os.path.join(data_dir, 'val')
        self.transform = transforms.Compose([transforms.ToTensor()])

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

        self.grid_height = target_size // stride
        self.grid_width = target_size // stride

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.imgs_paths[index])
        img = Image.open(img_path).convert('RGB')
        orig_size = img.size[0]
        img = self.transform(img)
        img, pad_w, pad_h, new_w, new_h = resize_and_pad_image(img, target_size=self.target_size)

        image_id = self.filename_to_id.get(self.imgs_paths[index])
        annotations = self.id_to_annotations.get(image_id, [])
        
        # initialization if three tensor maps
        regression_map = torch.zeros((4, self.grid_height, self.grid_width)) # (c , h , w)
        classification_map = torch.zeros((self.num_classes, self.grid_height, self.grid_width))
        centerness_map = torch.zeros((1, self.grid_height, self.grid_width)) # (c ,h ,w)
        
        # extraction of gt bbox and category id
        for anno in annotations:
            bbox = np.array(anno['bbox'])
            category_id = anno['category_id']
            if new_w < self.target_size or new_h < self.target_size:
                bbox = adjust_bbox_with_padding(bbox, pad_w, pad_h)
            # change to xyxy coordinate , and center coordinates of bbox, and get the coordinates on the final feature map (32 x 32)
            grid_x0, grid_y0, grid_x1, grid_y1, grid_cx, grid_cy = normalize_bbox(bbox, orig_size, self.target_size, self.stride)#check if cx and cy acan be removed
        # classification mapping
            classification_map[category_id , grid_y0:grid_y1+1 , grid_x0:grid_x1+1] = 1
            # print(f"classification map {classification_map[category_id]}")
            # print(f"unique values {torch.unique(classification_map)}")
            # print(f"classification map {classification_map.shape}")
            
        # Regression map 
            # 512 x 512 images --> 32 x 32 feature map --> each grid is 16 x 16 pixels
            _,h,w = img.shape
            h , w = h // self.stride , w // self.stride
            for y in range(grid_y0, grid_y1 + 1):
                for x in range(grid_x0, grid_x1 + 1):
                    l, t, r, b = distanceaway(x, y, grid_x0, grid_y0, grid_x1, grid_y1)
                    if regression_map[:, y, x].sum() == 0:  # If cell is empty, assign directly
                        regression_map[:, y, x] = torch.tensor([l, t, r, b])
                    else:
                        # If there is already a value, choose the box with the smaller area
                        existing_l, existing_t, existing_r, existing_b = regression_map[:, y, x]
                        existing_area = (existing_l + existing_r) * (existing_t + existing_b)
                        new_area = (l + r) * (t + b)
                        if new_area < existing_area:
                            regression_map[:, y, x] = torch.tensor([l, t, r, b])
                    
                    # centerness map
                    centerness = calculate_centerness(l, t, r, b)
                    centerness_map[:, y, x] = centerness
            

        return img, regression_map, classification_map , centerness_map








'''
if 0 <= grid_cx < self.grid_width and 0 <= grid_cy < self.grid_height:
                if classification_map[category_id, grid_cy, grid_cx] == 0:
                    classification_map[category_id, grid_cy, grid_cx] = 1
                    regression_map[:, grid_cy, grid_cx] = torch.tensor([l, t, r, b])
                    centerness_map[0, grid_cy, grid_cx] = np.sqrt((min(l, r) / max(l, r)) * (min(t, b) / max(t, b)))
                else:
                    existing_regression = regression_map[:, grid_cy, grid_cx]
                    existing_l, existing_t, existing_r, existing_b = existing_regression
                    existing_cx = (grid_cx + 0.5) * self.stride
                    existing_cy = (grid_cy + 0.5) * self.stride
                    existing_x0 = existing_cx - existing_l * self.stride
                    existing_y0 = existing_cy - existing_t * self.stride
                    existing_x1 = existing_cx + existing_r * self.stride
                    existing_y1 = existing_cy + existing_b * self.stride
                    existing_area = (existing_x1 - existing_x0) * (existing_y1 - existing_y0)
                    new_area = bbox[2] * bbox[3]

                    if new_area < existing_area:
                        classification_map[category_id, grid_cy, grid_cx] = 1
                        regression_map[:, grid_cy, grid_cx] = torch.tensor([l, t, r, b])
                        centerness_map[0, grid_cy, grid_cx] = np.sqrt((min(l, r) / max(l, r)) * (min(t, b) / max(t, b)))
'''

                
            
            
        
            
            