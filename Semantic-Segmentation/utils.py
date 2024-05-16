import random
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.morphology import remove_small_objects, remove_small_holes 
from skimage.segmentation import watershed
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy import ndimage
from skimage.feature import peak_local_max
import numpy as np
import pandas as pd
from skimage.morphology import dilation,square,erosion
from skimage.segmentation import watershed
from skimage.measure import label
from PIL import Image,ImageDraw
import pandas as pd
from shapely.geometry import shape
from shapely.wkt import dumps
from shapely.ops import unary_union

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
    # Convert tensor to numpy array
    outputs = outputs.cpu().numpy()

    # Direct operation on numpy array
    binary_masks = np.squeeze((outputs > threshold).astype(np.float32))

    # No need to convert to list if dimension is 3
    if binary_masks.ndim == 2:
        binary_masks = [binary_masks]

    processed_masks = []
    for binary_mask_np in binary_masks:
        # Morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)  # Smaller kernel
        closed_mask = cv2.morphologyEx(binary_mask_np, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Ensure labels are integer type
        labels_mask = closed_mask.astype(np.int32)

        # Apply the watershed algorithm
        distance = ndimage.distance_transform_edt(closed_mask)
        local_maxi = peak_local_max(distance, min_distance=1, exclude_border=False, footprint=np.ones((3, 3)), labels=labels_mask)

        # Convert peak locations to a boolean mask
        peaks = np.zeros_like(distance, dtype=bool)
        peaks[local_maxi[:, 0], local_maxi[:, 1]] = True

        markers = ndimage.label(peaks)[0]
        labels = watershed(-distance, markers, mask=closed_mask)

        # Convert labels to binary masks and remove small objects
        for label in np.unique(labels):
            if label == 0:
                continue
            label_mask = (labels == label).astype(np.uint8)
            label_mask = remove_small_objects(label_mask.astype(bool), min_size=min_size)
            label_mask = remove_small_holes(label_mask, area_threshold=min_size)

            # Apply bilateral filter for edge-preserving smoothing
            if smoothing:
                label_mask = cv2.bilateralFilter(label_mask.astype(np.float32), d=5, sigmaColor=50, sigmaSpace=50)
                _, label_mask = cv2.threshold(label_mask, 0.5, 1, cv2.THRESH_BINARY)

            processed_masks.append(label_mask)
    
    # Convert the list of numpy arrays to a single numpy array, then to a tensor
    return torch.from_numpy(np.array(processed_masks))

def plot(model, images, gt_masks, checkpoint, device , with_postprocess = True):
    # Load model weights
    model.load_state_dict(torch.load(checkpoint)['model_state_dict'])
    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(images.to(device))
        outputs = torch.sigmoid(outputs)  # Assuming binary classification (sigmoid output)
            # Convert model outputs to numpy arrays, then post-process
            # outputs = postprocess(outputs)
        if with_postprocess:
            postprocess = PostProcessing()
            outputs = outputs.cpu().numpy()
            outputs = postprocess.post_process_batch(outputs)
            # outputs = postprocess.noise_filter(outputs , mina = 10)
        outputs = torch.tensor(outputs)
    print("outputs shape", outputs.shape)   
    images = images.cpu().numpy()
    gt_masks = gt_masks.squeeze(1).cpu().numpy()  # Remove channel dim if it's 1
    outputs = outputs.squeeze(0)
    predicted_masks = outputs.permute(0 , 2,3,1).cpu().numpy()  # Thresholding to create binary mask

    num_images = images.shape[0]
    fig, axs = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    
    for idx in range(num_images):
        axs[idx, 0].imshow(np.transpose(images[idx], (1, 2, 0)))  # Convert from CHW to HWC format
        axs[idx, 0].set_title('Input Image')
        axs[idx, 1].imshow(gt_masks[idx], cmap='viridis')
        axs[idx, 1].set_title('Ground Truth Mask')
        axs[idx, 2].imshow(predicted_masks[idx], cmap='viridis')
        axs[idx, 2].set_title('Predicted Mask')

        for ax in axs[idx]:
            ax.axis('off')
    
    plt.tight_layout()
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
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    thresholds = [0.3, 0.5, 0.75]
    iou_scores = []
    for threshold in thresholds:
        preds_binary = np.array([(pred > threshold).astype(np.float32) for pred in preds])
        labels_binary = np.array([(label == pos_label).astype(np.float32) for label in labels])

        # Calculate intersection and union
        intersection = np.sum(preds_binary * labels_binary)
        union = np.sum(preds_binary) + np.sum(labels_binary) - intersection

        # Compute IoU or handle the case with no presence of the class
        if np.count_nonzero(union) == 0:
            iou_scores.append(1.0 if np.count_nonzero(intersection) == 0 else 0.0)
        else:
            iou_scores.append(intersection / union)

    # Unpack the list to return individual IoU scores
    return tuple(iou_scores)


def calc_f1_score(predicted_masks, gt_masks, threshold=0.5):
    # Flatten the masks
    # predicted_masks = torch.sigmoid(predicted_masks)
    predicted_masks = predicted_masks.cpu().numpy()
    gt_masks = gt_masks.cpu().numpy()
    predicted_masks = (predicted_masks > threshold).astype(np.float32).flatten()
    gt_masks = gt_masks.flatten()
    # print(gt_masks.shape, predicted_masks.shape)
    # Calculate precision, recall, and f1 score
    precision = precision_score(gt_masks, predicted_masks, average='binary' , zero_division= 1)
    recall = recall_score(gt_masks, predicted_masks, average='binary' , zero_division = 1)
    f1 = f1_score(gt_masks, predicted_masks, average='binary' , zero_division = 1)
    
    return precision, recall, f1

def dice_loss(predicted_masks, gt_masks, smooth=1e-6):
    # Apply sigmoid to get probabilities
    # predicted_masks = torch.sigmoid(predicted_masks)
    
    # Flatten the masks
    predicted_masks = predicted_masks.to('cuda')
    predicted_masks = predicted_masks.flatten()
    gt_masks = gt_masks.flatten()
    
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






class PostProcessing():
    def post_process(self, pred, thresh=0.5, thresh_b=0.6, mina=100, mina_b=50):
        """
        Post-processes a prediction mask to obtain a refined segmentation.This function refines 
        a semantic segmentation mask, particularly for building segmentation tasks.
        It leverages optional channels for predicting borders and spacing around buildings.

        Args:
            pred (numpy.ndarray): Prediction mask with shape (height, width, channels).
            thresh (float, default=0.5): Threshold for considering pixels as part of the final segmentation.
            thresh_b (float, default=0.6): Threshold for considering pixels as borders between objects.
            mina (int, default=100): Minimum area threshold for retaining segmented regions.
            mina_b (int, default=50): Minimum area threshold for retaining basins.

        Returns:
            numpy.ndarray: Refined segmentation mask.

        Description:
            The refinement process involves:
            1. Extracting individual channels from the input mask.
            2. Separating nuclei (building interiors) from borders based on the predicted borders channel.
            3. Applying thresholding to identify basins within nuclei, which represent potential individual buildings.
            4. Optionally filtering out small basins based on the minimum area threshold.
            5. Performing watershed segmentation to separate closely located buildings.
            6. Applying noise filtering to remove small regions from the segmented mask.
            7. Returning the refined segmentation mask with labeled and filtered regions.

        Note:
            - The function assumes a specific order for input channels:
                - Channel 0: Building predictions
                - Channel 1 (optional): Border predictions
                - Channel 2 (optional): Spacing predictions
            - The output represents labeled regions in the refined segmentation.

        """
        if len(pred.shape) < 2:
            return None
        if len(pred.shape) == 2:
            pred = pred[..., np.newaxis]

        ch = pred.shape[2]
        buildings = pred[..., 0]

        if ch > 1:
            borders = pred[..., 1]
            nuclei = buildings * (1.0 - borders)

            if ch == 3:
                spacing = pred[..., 2]
                nuclei *= (1.0 - spacing)

            basins = label(nuclei > thresh_b, background=0, connectivity=2)

            if mina_b > 0:
                basins = self.noise_filter(basins, mina=mina_b)
                basins = label(basins, background=0, connectivity=2)

            washed = watershed(image=-buildings, markers=basins, mask=buildings > thresh, watershed_line=False)

        elif ch == 1:
            washed = buildings > thresh

        washed = label(washed, background=0, connectivity=2)
        washed = self.noise_filter(washed, mina=mina)
        washed = label(washed, background=0, connectivity=2)

        return washed
    
    @staticmethod
    def noise_filter(washed,mina):
        """
        Filter small regions in a labeled segmentation mask based on minimum area.
        This function filters out small labeled regions in a segmentation mask based on their area.
        It iterates over unique labels, calculates the area for each label, and sets the pixel values
        corresponding to labels with area less than or equal to the specified threshold to 0.

        Args:
            washed (numpy.ndarray): Input labeled segmentation mask.
            mina (int): Minimum area threshold for retaining labeled regions.

        Returns:
            numpy.ndarray: Segmentation mask with small labeled regions filtered out.
        """
        values = np.unique(washed)
        for val in values[1:]:
            area = (washed[washed == val]>0).sum()
            if area<=mina:
                washed[washed == val] = 0
        return washed
    
    def post_process_batch(self, preds, thresh=0.5, thresh_b=0.6, mina=100, mina_b=50):
        processed_preds = []
        for pred in preds:
            processed_pred = self.post_process(pred.squeeze(0), thresh, thresh_b, mina, mina_b)
            processed_pred = self.noise_filter(processed_pred, mina=100)
            # Add an extra dimension to match the shape of the ground truth masks
            processed_pred = np.expand_dims(processed_pred, axis=0)
            processed_preds.append(processed_pred)
        return np.array(processed_preds)
