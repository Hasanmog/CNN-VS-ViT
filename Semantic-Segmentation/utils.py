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


def calculate_iou(preds, labels, pos_label=1):
    """
    Calculate the Intersection over Union (IoU) for a single class segmentation
    where predictions are given as sigmoid outputs at specific thresholds (0.3, 0.5, 0.75).

    Args:
        preds (torch.Tensor): Logits output from the model (N, C, H, W), where
                              N is the batch size, C is the number of classes,
                              H and W are the height and width of the image.
        labels (torch.Tensor): Ground truth labels of shape (N, H, W), with values 0 and 1.
        pos_label (int): The label of the positive class (default is 1).

    Returns:
        tuple: IoU scores for the positive class at thresholds 0.3, 0.5, and 0.75.
    """
    thresholds = [0.3, 0.5, 0.75]
    iou_scores = []
    probs = torch.sigmoid(preds)  # Convert logits to probabilities

    for threshold in thresholds:
        # Apply threshold to convert probabilities to binary predictions
        preds_binary = (probs > threshold).float()  # Use float for binary conversion

        # Ensure labels are also boolean tensors if not already
        labels_binary = (labels == pos_label).float()  # Use float here as well

        # Calculate intersection and union
        intersection = (preds_binary * labels_binary).sum()  # Use element-wise multiplication
        union = preds_binary.sum() + labels_binary.sum() - intersection

        # Compute IoU or handle the case with no presence of the class
        if union == 0:
            iou_scores.append(1.0 if intersection == 0 else 0.0)
        else:
            iou_scores.append((intersection / union).item())

    # Unpack the list to return individual IoU scores
    return tuple(iou_scores)

