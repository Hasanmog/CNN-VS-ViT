import random
import os
import torch
import numpy as np


def dataset_split(img_dir, masks_dir):
    # List all image files
    images = os.listdir(img_dir)
    random.seed(42)  # Set seed for reproducibility
    random.shuffle(images)  # Shuffle images

    # Split into train, test, and validation sets
    train = images[:280]
    test = images[280:340]
    val = images[340:]

    sets = [train, test, val]
    train_masks = []
    test_masks = []
    val_masks = []
    masks_sets = [train_masks, test_masks, val_masks]

    # List all mask files
    masks = os.listdir(masks_dir)
    mask_dict = {mask.split('.')[0]: mask for mask in masks}  # Create a dictionary with numerical part as key

    # Match masks to images in each set
    for i, img_set in enumerate(sets):
        for image in img_set:
            num_id = image.split('.')[0]  # Extract the numerical ID from the image filename
            corresponding_mask = mask_dict.get(num_id, None)  # Get the corresponding mask using the numerical ID
            if corresponding_mask:
                masks_sets[i].append(corresponding_mask)

    return sets, masks_sets


def calculate_iou(preds, labels, threshold=0.5, pos_label=1):
    """
    Calculate the Intersection over Union (IoU) for a single class segmentation
    where predictions are given as sigmoid outputs.

    Args:
        preds (torch.Tensor): Sigmoid predictions of shape (N, C, H, W) where
                              N is the batch size, C is the number of classes (typically 2 for binary classification),
                              H and W are the height and width of the image.
        labels (torch.Tensor): Ground truth labels of shape (N, H, W), with values 0 and 1.
        threshold (float): Threshold to convert sigmoid outputs to binary class labels.
        pos_label (int): The label of the positive class (default is 1).

    Returns:
        float: IoU for the positive class.
    """
    # Apply threshold to sigmoid predictions to convert to binary class labels
    preds = preds > threshold  # This creates a binary tensor of 0's and 1's

    # Ensure labels are also boolean tensors if not already
    labels = labels == pos_label

    # Calculate intersection and union
    intersection = (preds & labels).float().sum()  # Logical AND
    union = (preds | labels).float().sum()         # Logical OR

    if union == 0:
        iou = float('nan')  # Handle case where there is no presence of the class in both pred and labels
    else:
        iou = (intersection / union).item()  # Convert to float if not zero

    return iou

