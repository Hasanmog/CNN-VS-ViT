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

def calculate_accuracy(cls_pred, cls_true):
    cls_pred = torch.sigmoid(cls_pred) > 0.5  # Assuming binary classification with sigmoid
    correct = (cls_pred == cls_true).sum().item()
    total = cls_true.numel()
    return correct / total


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two sets of boxes.
    box1 and box2 are expected to be of shape [N, 4] with each row in format (x1, y1, x2, y2)
    Returns the IoU for each pair of boxes.
    """
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])
    
    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area
    return iou


def calculate_precision_recall(pred_boxes, true_boxes, iou_threshold):
    ious = calculate_iou(pred_boxes, true_boxes)
    true_positives = (ious > iou_threshold).sum().item()
    precision = true_positives / pred_boxes.size(0)
    recall = true_positives / true_boxes.size(0)
    return precision, recall




