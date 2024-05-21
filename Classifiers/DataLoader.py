import pandas as pd
import os
import re
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
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
    
    
def tif2png(set , out_dir , format = 'PNG'):
    """
    Function for converting .tif images to .png
    Arguments:
    set : list of paths to .tif images
    out_dir : path to save the converted images
    format : format to convert to (default is PNG)
    Returns:
    None
    """
    for idx,img in enumerate(set):
        image = Image.open(img)
        filename , _ = os.path.splitext(img) # remove .tif
        image.save(os.path.join(out_dir, f'{os.path.basename(filename)}.{format.lower()}'), format) # replacing .format 
        print(f"Done converting image {idx} to .png")

def split(parent_dir):
    """
    This function splits the dataset into train, validation and test sets
    
    Arguments:
    parent_dir : path to the folder with the unsplitted dataset
    
    returns:
    train , validation and test sets 
    Note : using train 70% , validation 15% and test 15% split
    """
    classes = os.listdir(parent_dir)
    class_sets = [os.path.join(parent_dir, i) for i in classes]  # path to each class folder
    print(f"length of dataset is {len(class_sets)}")
    
    train = []
    val = []
    test = []
    
    for category in class_sets:
        imgs = os.listdir(category)
        
        # Calculate split indices
        num_imgs = len(imgs)
        train_end = int(0.7 * num_imgs)
        val_end = int(0.85 * num_imgs)
        
        # Create lists of paths for each split
        train_imgs = [os.path.join(category, img) for img in imgs[:train_end]]
        val_imgs = [os.path.join(category, img) for img in imgs[train_end:val_end]]
        test_imgs = [os.path.join(category, img) for img in imgs[val_end:]]
        
        
        train.extend(train_imgs)
        val.extend(val_imgs)
        test.extend(test_imgs)
    
    return train, val, test  


def extract_label(img_name):
    # Split the filename on the first digit and take the first part
    label = re.split(r'(\d+)', img_name)[0]
    # Add a space between the words
    label = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', label)
    return label


def custom_collate_fn(batch):
    class_to_idx = {
        'agricultural': 0, 'airplane': 1, 'baseballdiamond': 2, 'beach': 3,
        'buildings': 4, 'chaparral': 5, 'denseresidential': 6, 'forest': 7,
        'freeway': 8, 'golfcourse': 9, 'harbor': 10, 'intersection': 11,
        'mediumresidential': 12, 'mobilehomepark': 13, 'overpass': 14,
        'parkinglot': 15, 'river': 16, 'runway': 17, 'sparseresidential': 18,
        'storagetanks': 19, 'tenniscourt': 20
    }
    images = [item[0] for item in batch]
    label_tuples = [item[1] for item in batch]

    images = default_collate(images)
    labels = []

    for label_tuple in label_tuples:
        # Check if label_tuple is actually a string
        if isinstance(label_tuple, str):
            label_tuple = [label_tuple]  # Convert single string to list

        label_indices = []
        for label in label_tuple:
            try:
                label_indices.append(class_to_idx[label])
            except KeyError:
                raise ValueError(f"Label '{label}' not found in class index dictionary.")

        labels.append(torch.tensor(label_indices))

    labels = torch.cat(labels, dim=0)
    return images, labels



class UC_MERCED(Dataset):
    def __init__(self, parent_dir , transform):
        
        """
        parent_dir : path to each set
        """
        self.parent_dir = parent_dir
        self.img_paths = os.listdir(self.parent_dir)
        self.labels = [extract_label(img) for img in self.img_paths]
        self.transform = transform
    
    def __len__(self ):
        return len(self.labels)  
    
    def __getitem__(self, index):
        
        image_path = os.path.join(self.parent_dir , self.img_paths[index])
        img = Image.open(image_path)
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
            
        return img , label      

    
        
    
    
        
    