import numpy as np
import torch
import math 
from torchvision.ops import box_iou
import itertools

def file_format_counter(imgs_paths):
    '''
    simple function to count the number of images corresponding to each of the following extensions : .png , .jpg and .bmp
    Arguments:
    imgs_paths : list of the imgs paths
    returns : number of .png , .jpg and .bmp images in this list
    '''
    png , jpg , bmp = 0 , 0 , 0
    for index,img in enumerate(imgs_paths):
        if img.endswith('.png'):
            png+=1
        elif img.endswith(".jpg"):
            jpg+=1
        else:
            bmp +=1
            
    return png , jpg , bmp


def bbox_iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    if isinstance(box1, torch.Tensor):
        box1 = box1.tolist()
    if isinstance(box2, torch.Tensor):
        box2 = box2.tolist()

    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

# CHECK WHICH IS USEFUL TO KEEP AND REMOVE OTHERS

def normalize_bboxes(gt_boxes, img_width, img_height):
    """
    Normalize bounding boxes. Assumes boxes are in the format [x_center, y_center, width, height].
    
    Parameters:
        gt_boxes (list of lists): The bounding boxes for each image in the batch.
        img_width (int or float): Width of the image.
        img_height (int or float): Height of the image.
    
    Returns:
        list of list: Normalized bounding boxes.
    """
    normalized_boxes = []
    for boxes in gt_boxes:
        norm_boxes = []
        for box in boxes:
            x_center, y_center, width, height = box
            # Normalize center coordinates and dimensions
            x_center /= img_width
            y_center /= img_height
            width /= img_width
            height /= img_height
            norm_boxes.append([x_center, y_center, width, height])
        normalized_boxes.append(norm_boxes)
    return torch.tensor(normalized_boxes)




