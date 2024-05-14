import random
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.morphology import remove_small_objects, remove_small_holes
from sklearn.metrics import precision_score, recall_score, f1_score

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


def postprocess(outputs, threshold=0.5, min_size=500, smoothing=True):
    """
    used to clean the output return from the model
    
    Args:
    outputs : model output after sigmoid
    """
    
    binary_masks = (outputs > threshold).float().cpu().numpy().squeeze()

    if binary_masks.ndim == 3:
        binary_masks = [binary_masks[i] for i in range(binary_masks.shape[0])]
    elif binary_masks.ndim == 2:
        binary_masks = [binary_masks]

    processed_masks = []
    for binary_mask_np in binary_masks:
        # Morphological closing to fill small holes
        kernel = np.ones((5, 5), np.uint8)
        closed_mask = cv2.morphologyEx(binary_mask_np, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Morphological opening to remove noise
        opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Remove small objects
        cleaned_mask = remove_small_objects(opened_mask.astype(bool), min_size=min_size)
        cleaned_mask = remove_small_holes(cleaned_mask, area_threshold=min_size)
        cleaned_mask = cleaned_mask.astype(np.uint8)

        # Apply bilateral filter for edge-preserving smoothing
        if smoothing:
            cleaned_mask = cv2.bilateralFilter(cleaned_mask.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75)
            _, cleaned_mask = cv2.threshold(cleaned_mask, 0.5, 1, cv2.THRESH_BINARY)
        
        processed_masks.append(cleaned_mask)
    
    return processed_masks

def plot(model, images, gt_masks,  device , checkpoint = None,):
    """
    This function is used to plot a batch of images, ground truth masks and the predictions of a model
    Args:
    model : Segmentation model
    image : Batch of images from the dataloader
    gt_masks : Batch of ground truth masks from the dataloader
    checkpoint : optional , path to the model checkpoint to load
    device : device to use for computations
    Returns:
    plot showing images vs gt_masks vs pred_masks 
    """
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        outputs = model(images.to(device))
        outputs = torch.sigmoid(outputs)
    outputs = postprocess(outputs)

    columns = 3
    rows = len(outputs)
    fig, axs = plt.subplots(nrows=rows, ncols=columns, figsize=(15, rows * 5))
    plt.subplots_adjust(wspace=0, hspace=0.1)
    
    for i, (img, gt_mask, out) in enumerate(zip(images, gt_masks, outputs)):
        img = img.permute(1, 2, 0).numpy()
        gt_mask = gt_mask.squeeze().numpy()
         
        axs[i][0].imshow(img)
        axs[i][0].set_title("Image")
        axs[i][0].set_axis_off()
            
        axs[i][1].imshow(gt_mask, cmap='viridis')
        axs[i][1].set_title("Ground Truth")
        axs[i][1].set_axis_off()
            
        axs[i][2].imshow(out, cmap='viridis')
        axs[i][2].set_title("Predicted mask")
        axs[i][2].set_axis_off()
            
    plt.show()
    
    
def iou(preds, labels, pos_label=1):
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


def calc_f1_score(predicted_masks, gt_masks, threshold=0.5):
    # Flatten the masks
    predicted_masks = torch.sigmoid(predicted_masks)
    predicted_masks = (predicted_masks > threshold).float().cpu().numpy().flatten()
    gt_masks = gt_masks.cpu().numpy().flatten()
    
    # Calculate precision, recall, and f1 score
    precision = precision_score(gt_masks, predicted_masks, average='binary' , zero_division= 1)
    recall = recall_score(gt_masks, predicted_masks, average='binary' , zero_division = 1)
    f1 = f1_score(gt_masks, predicted_masks, average='binary' , zero_division = 1)
    
    return precision, recall, f1

def dice_loss(predicted_masks, gt_masks, smooth=1e-6):
    # Apply sigmoid to get probabilities
    predicted_masks = torch.sigmoid(predicted_masks)
    
    # Flatten the masks
    predicted_masks = predicted_masks.view(-1)
    gt_masks = gt_masks.view(-1)
    
    # Compute intersection and sums
    intersection = (predicted_masks * gt_masks).sum()
    total = (predicted_masks + gt_masks).sum()
    
    # Compute Dice coefficient
    dice_coefficient = (2.0 * intersection + smooth) / (total + smooth)
    
    # Compute Dice Loss
    dice_loss = 1.0 - dice_coefficient
    
    return dice_loss


def IoU(maskA, maskB):
    # Ensure the masks are on cpu
    maskA = maskA.cpu()
    maskB = maskB.cpu()

    # Convert the masks to numpy arrays
    maskA = maskA.numpy()
    maskB = maskB.numpy()

    intersection = np.logical_and(maskA, maskB)
    union = np.logical_or(maskA, maskB)
    iou = np.sum(intersection) / np.sum(union)
    return iou

class matching_algorithm:
    def __init__(self, gt_masks, pred_masks, iou_threshold=0.5):
        self.gt_masks = gt_masks
        self.pred_masks = pred_masks
        self.iou_threshold = iou_threshold

    def matching(self):
        if len(self.pred_masks) == 0 or len(self.gt_masks) == 0:
            print("Both predicted and ground truth masks are empty.")
            return [], [], [], [], [], []

        iou_matrix = np.zeros((len(self.pred_masks), len(self.gt_masks)))

        for i in range(len(self.pred_masks)):
            for j in range(len(self.gt_masks)):
                iou_matrix[i, j] = IoU(self.pred_masks[i], self.gt_masks[j])

        iou_list = []
        f1_scores = []
        pred_matched = set()
        gt_matched = set()
        tp_pred_indices = []
        tp_gt_indices = []

        while True:
            max_iou = np.max(iou_matrix)
            if max_iou < self.iou_threshold:
                break
            max_index = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            iou_list.append(max_iou)
            pred_matched.add(max_index[0])
            gt_matched.add(max_index[1])

            tp_pred_indices.append(max_index[0])
            tp_gt_indices.append(max_index[1])

            f1_score = 2 * max_iou / (max_iou + 1)
            f1_scores.append(f1_score)

            print(f"Matched predicted mask {max_index[0]} with GT mask {max_index[1]}, IoU = {max_iou}, F1 = {f1_score}")
            iou_matrix[max_index[0], :] = 0
            iou_matrix[:, max_index[1]] = 0

        for i in set(range(len(self.pred_masks))) - pred_matched:
            iou_list.append(0)
            f1_scores.append(0)
            print(f"Unmatched predicted mask {i} has no match, IoU = 0, F1 = 0")

        for i in set(range(len(self.gt_masks))) - gt_matched:
            iou_list.append(0)
            f1_scores.append(0)
            print(f"Unmatched GT mask {i} has no match, IoU = 0, F1 = 0")

        print("number of GT masks:", len(self.gt_masks))
        print("number of predicted masks:", len(self.pred_masks))

        fp_indices = list(set(range(len(self.pred_masks))) - pred_matched)
        fn_indices = list(set(range(len(self.gt_masks))) - gt_matched)

        return iou_list, f1_scores, tp_pred_indices, tp_gt_indices, fp_indices, fn_indices

    def tp_iou(self, tp_pred_indices, tp_gt_indices):
        tp_iou_list = []
        for i, j in zip(tp_pred_indices, tp_gt_indices):
            iou = IoU(self.pred_masks[i], self.gt_masks[j])
            tp_iou_list.append(np.nan_to_num(iou))

        if len(tp_iou_list) > 0:
            avg_tp_iou = np.mean(tp_iou_list)
        else:
            avg_tp_iou = 0
        return tp_iou_list, avg_tp_iou

