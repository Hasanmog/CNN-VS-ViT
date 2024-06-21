import numpy as np
import torch
import math 
from torchvision.ops import box_iou
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

def calculate_accuracy(pred_cls, true_cls):
    pred_cls = torch.argmax(pred_cls, dim=1)
    true_cls = torch.argmax(true_cls, dim=1)
    correct = (pred_cls == true_cls).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy.item()

def visualize_centerness_on_image(image, centerness_map, stride):
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap='gray')

    # Assume centerness_map has shape [1, height, width]
    height, width = centerness_map.shape[1], centerness_map.shape[2]

    for y in range(height):
        for x in range(width):
            centerness_value = centerness_map[0, y, x].item()  # Convert tensor to float
            if centerness_value > 0:  # Only plot non-zero values
                cx = (x + 0.5) * stride
                cy = (y + 0.5) * stride
                
                rect = patches.Rectangle((cx - stride / 2, cy - stride / 2), stride, stride, 
                                         linewidth=1, edgecolor='r', facecolor='none', alpha=centerness_value)
                ax.add_patch(rect)

    plt.show()
    
def visualize_bboxes_on_image(image, regression_map, stride):
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap='gray')

    height, width = regression_map.shape[1], regression_map.shape[2]

    for y in range(height):
        for x in range(width):
            l, t, r, b = regression_map[:, y, x]

            if l != 0 or t != 0 or r != 0 or b != 0:  # Only plot non-zero values
                cx = (x + 0.5) * stride
                cy = (y + 0.5) * stride
                
                x0 = cx - l * stride
                y0 = cy - t * stride
                box_w = (l + r) * stride
                box_h = (t + b) * stride
                
                rect = patches.Rectangle((x0, y0), box_w, box_h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

    plt.show()


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

def calculate_iou(box1, box2):
    x1 = torch.max(box1[:, None, 0], box2[:, 0])
    y1 = torch.max(box1[:, None, 1], box2[:, 1])
    x2 = torch.min(box1[:, None, 2], box2[:, 2])
    y2 = torch.min(box1[:, None, 3], box2[:, 3])

    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union_area = box1_area[:, None] + box2_area - inter_area

    iou = inter_area / union_area
    return iou


def calculate_precision_recall(pred_boxes, gt_boxes, iou_threshold=0.5):
    ious = calculate_iou(pred_boxes, gt_boxes)
    
    true_positives = (ious > iou_threshold).sum().item()
    total_pred_boxes = pred_boxes.size(0)
    total_true_boxes = gt_boxes.size(0)

    precision = true_positives / total_pred_boxes if total_pred_boxes > 0 else 0
    recall = true_positives / total_true_boxes if total_true_boxes > 0 else 0

    return precision, recall

def extract_bounding_boxes(pred, ground_truth):
    """
    Extract bounding boxes from prediction and ground truth tensors.
    
    Args:
    - pred (torch.Tensor): Predicted regression map with shape [batch_size, num_channels, height, width]
    - ground_truth (torch.Tensor): Ground truth regression map with the same shape as pred
    
    Returns:
    - pred_boxes (torch.Tensor): Reshaped predicted bounding boxes
    - ground_truth_boxes (torch.Tensor): Reshaped ground truth bounding boxes
    """
    # Assuming pred and ground_truth are in the format [batch_size, num_channels, height, width]
    batch_size = pred.size(0)
    num_boxes = pred.size(2) * pred.size(3)  # Total number of boxes in the height x width grid
    pred_boxes = pred.permute(0, 2, 3, 1).contiguous().view(batch_size, num_boxes, 4)
    ground_truth_boxes = ground_truth.permute(0, 2, 3, 1).contiguous().view(batch_size, num_boxes, 4)
    
    return pred_boxes, ground_truth_boxes
