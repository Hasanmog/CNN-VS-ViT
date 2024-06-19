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


def regression_map_to_boxes(regression_maps, stride):
    """
    Convert batched regression map values to bounding box coordinates.
    """
    batch_size, _, h, w = regression_maps.shape
    boxes = []

    for batch_idx in range(batch_size):
        batch_boxes = []
        for y in range(h):
            for x in range(w):
                l, t, r, b = regression_maps[batch_idx, :, y, x]
                cx = (x + 0.5) * stride
                cy = (y + 0.5) * stride
                x1 = cx - l * stride
                y1 = cy - t * stride
                x2 = cx + r * stride
                y2 = cy + b * stride
                batch_boxes.append([x1, y1, x2, y2])
        boxes.append(batch_boxes)

    return boxes

def calculate_iou(pred_boxes, gt_boxes):
    """
    Calculate the Intersection over Union (IoU) between batches of predicted and ground truth bounding boxes.
    pred_boxes and gt_boxes are expected to be of shape [batch_size, N, 4] with each box in format (x1, y1, x2, y2)
    Returns the IoU for each pair of boxes in the batch.
    """
    pred_boxes = pred_boxes.to('cpu')
    gt_boxes = gt_boxes.to('cpu')
    batch_size = pred_boxes.size(0)
    ious = []
    for i in range(batch_size):
        box1 = pred_boxes[i].clone().detach()
        box2 = gt_boxes[i].clone().detach()

        x1 = torch.max(box1[:, 0].unsqueeze(1), box2[:, 0].unsqueeze(0))
        y1 = torch.max(box1[:, 1].unsqueeze(1), box2[:, 1].unsqueeze(0))
        x2 = torch.min(box1[:, 2].unsqueeze(1), box2[:, 2].unsqueeze(0))
        y2 = torch.min(box1[:, 3].unsqueeze(1), box2[:, 3].unsqueeze(0))

        inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union_area = box1_area.unsqueeze(1) + box2_area.unsqueeze(0) - inter_area

        iou = inter_area / union_area
        ious.append(iou)

    return torch.stack(ious)


def calculate_precision_recall(pred_boxes, true_boxes, iou_threshold):
    """
    Calculate precision and recall for batches of predicted and ground truth boxes.
    """
    ious = calculate_iou(pred_boxes, true_boxes)
    true_positives = 0
    total_pred_boxes = 0
    total_true_boxes = 0
    
    for i in range(len(ious)):
        true_positives += (torch.tensor(ious[i]) > iou_threshold).sum().item()
        total_pred_boxes += len(pred_boxes[i])
        total_true_boxes += len(true_boxes[i])
    
    precision = true_positives / total_pred_boxes if total_pred_boxes > 0 else 0
    recall = true_positives / total_true_boxes if total_true_boxes > 0 else 0
    return precision, recall

def extract_bounding_boxes(pred, ground_truth):
    # Assuming pred and ground_truth are in the format [batch_size, num_channels, height, width]
    # Convert the feature maps to bounding box coordinates
    # This is an example and will depend on your specific model's output
    batch_size = pred.size(0)
    num_boxes = pred.size(2) * pred.size(3)  # Example, assuming each grid cell predicts one box
    pred_boxes = pred.permute(0, 2, 3, 1).contiguous().view(batch_size, num_boxes, 4)
    ground_truth_boxes = ground_truth.permute(0, 2, 3, 1).contiguous().view(batch_size, num_boxes, 4)
    return pred_boxes, ground_truth_boxes

