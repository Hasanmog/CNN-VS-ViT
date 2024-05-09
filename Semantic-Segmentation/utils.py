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

def calculate_iou(preds, labels, num_classes = 23):
    """
    Calculate the Intersection over Union (IoU) from model predictions after softmax.

    Args:
        preds (torch.Tensor): Softmaxed predictions of shape (N, C, H, W) where
                              N is the batch size, C is the number of classes,
                              H and W are the height and width of the image.
        labels (torch.Tensor): Ground truth labels of shape (N, H, W).
        num_classes (int): Number of classes.

    Returns:
        float: Mean IoU over all classes.
    """
    # Convert probabilistic predictions to discrete class predictions
    preds = torch.argmax(preds, dim=1)  # This collapses the C (classes) dimension

    ious = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        label_inds = (labels == cls)
        intersection = (pred_inds & label_inds).float().sum()  # Logical AND
        union = (pred_inds | label_inds).float().sum()         # Logical OR
        
        if union == 0:
            ious.append(float('nan'))  # No presence of class in both pred and label
        else:
            ious.append((intersection / union).item())

    mean_iou = np.nanmean(ious)  # Compute the mean IoU, ignoring NaN values
    return mean_iou

