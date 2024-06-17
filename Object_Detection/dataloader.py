import json
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def resize(img, target_size=512):
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

    return img

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
        img = resize(img, target_size=self.target_size)

        image_id = self.filename_to_id.get(self.imgs_paths[index])
        annotations = self.id_to_annotations.get(image_id, [])

        regression_map = torch.zeros((4, self.grid_height, self.grid_width))
        classification_map = torch.zeros((self.num_classes, self.grid_height, self.grid_width))
        centerness_map = torch.zeros((1, self.grid_height, self.grid_width))

        for anno in annotations:
            bbox = np.array(anno['bbox'])
            category_id = anno['category_id']
            x0, y0, w, h = bbox
            x1 = x0 + w
            y1 = y0 + h
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            epsilon = 1e-6

            grid_x0 = int(x0 / orig_size * self.grid_width)
            grid_y0 = int(y0 / orig_size * self.grid_height)
            grid_x1 = int(x1 / orig_size * self.grid_width)
            grid_y1 = int(y1 / orig_size * self.grid_height)
            grid_cx = int(cx / orig_size * self.grid_width)
            grid_cy = int(cy / orig_size * self.grid_height)

            l = (cx - x0) / self.stride
            t = (cy - y0) / self.stride
            r = (x1 - cx) / self.stride
            b = (y1 - cy) / self.stride

            bbox_area = w * h

            # Check for overlapping bounding boxes and prioritize the smaller one
            if classification_map[category_id, grid_cy, grid_cx] == 0:
                classification_map[category_id, grid_cy, grid_cx] = 1
                regression_map[:, grid_cy, grid_cx] = torch.tensor([l, t, r, b])
                centerness_map[0, grid_cy, grid_cx] = np.sqrt((min(l, r) / max(l, r)) * (min(t, b) / max(t, b)))
            else:
                # Calculate the area of the existing bounding box
                existing_regression = regression_map[:, grid_cy, grid_cx]
                existing_l, existing_t, existing_r, existing_b = existing_regression
                existing_cx = (grid_cx + 0.5) * self.stride
                existing_cy = (grid_cy + 0.5) * self.stride
                existing_x0 = existing_cx - existing_l * self.stride
                existing_y0 = existing_cy - existing_t * self.stride
                existing_x1 = existing_cx + existing_r * self.stride
                existing_y1 = existing_cy + existing_b * self.stride
                existing_area = (existing_x1 - existing_x0) * (existing_y1 - existing_y0)

                # Replace if the new bounding box is smaller
                if bbox_area < existing_area:
                    classification_map[category_id, grid_cy, grid_cx] = 1
                    regression_map[:, grid_cy, grid_cx] = torch.tensor([l, t, r, b])
                    centerness_map[0, grid_cy, grid_cx] = np.sqrt((min(l, r) / max(l, r)) * (min(t, b) / max(t, b)))

        return img, regression_map, classification_map, centerness_map




                
            
            
        
            
            