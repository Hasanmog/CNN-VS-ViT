
import numpy as np
import cv2
import torch
from torchvision.ops import nms

# this is taken from https://gist.github.com/ciklista/7973ccf346ac7fd0a83e8ffa761af5de
def convert_to_mins_maxes(box_xywh, input_shape=torch.tensor([512, 512], device='cuda:0')):
    """
    Converts boxes of YOLO output from (x_center, y_center, w, h) to (y_min, x_min, y_max, x_max)
    directly using PyTorch on the GPU.
    """
    # Split the last dimension into xy and wh
    box_xy = box_xywh[..., :2]  # x_center, y_center
    box_wh = box_xywh[..., 2:]  # width, height

    # Reverse the order to yx and hw
    box_yx = box_xy.flip(dims=[-1])  # reverse xy to yx
    box_hw = box_wh.flip(dims=[-1])  # reverse wh to hw

    # Compute mins and maxes
    box_mins = (box_yx - (box_hw / 2.0)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.0)) / input_shape

    # Concatenate mins and maxes to get the final boxes in corner format
    boxes = torch.cat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], dim=-1)

    return boxes


# These are taken from : https://gist.github.com/juliussin/b021d0ae74f6b4700009636335e48755
def xyxy_to_xywh(xyxy):
    """
    Convert XYXY format (x,y top left and x,y bottom right) to XYWH format (x,y center point and width, height).
    :param xyxy: [X1, Y1, X2, Y2]
    :return: [X, Y, W, H]
    """
    if np.array(xyxy).ndim > 1 or len(xyxy) > 4:
        raise ValueError('xyxy format: [x1, y1, x2, y2]')
    x_temp = (xyxy[0] + xyxy[2]) / 2
    y_temp = (xyxy[1] + xyxy[3]) / 2
    w_temp = abs(xyxy[0] - xyxy[2])
    h_temp = abs(xyxy[1] - xyxy[3])
    return np.array([int(x_temp), int(y_temp), int(w_temp), int(h_temp)])


import torch

def xywh_to_xyxy(xywh):
    """
    Convert a batch of bounding boxes from XYWH format (center x, center y, width, height)
    to XYXY format (top-left x, top-left y, bottom-right x, bottom-right y) for an entire batch.
    
    :param xywh: Tensor of shape [batch_size, N, 4] where N is the number of boxes per image.
    :return: Tensor of the same shape with each box converted to XYXY format.
    """
    if xywh.dim() != 3 or xywh.size(2) != 4:
        raise ValueError('Input xywh tensor must be of shape [batch_size, N, 4]')
    
    # Extract the centers (x, y) and sizes (width, height)
    x_c = xywh[..., 0]
    y_c = xywh[..., 1]
    w = xywh[..., 2]
    h = xywh[..., 3]
    
    # Compute top-left (x1, y1) and bottom-right (x2, y2)
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    
    # Stack the coordinates in the last dimension
    return torch.stack((x1, y1, x2, y2), dim=-1)




def post_process_outputs(outputs, image_width = 512, image_height = 512):
    # Assuming outputs are [batch_size, grid_size, grid_size, num_anchors, (4 + 1 + num_classes)]
    # and the first four are bbox coordinates in the order [x_center, y_center, width, height]
    sig = torch.sigmoid
    outputs[..., :2] = sig(outputs[..., :2])  # Normalize x_center, y_center
    outputs[..., 2:4] = sig(outputs[..., 2:4])  # Normalize width, height
    
    # Convert from center to corner format
    x_center, y_center = outputs[..., 0], outputs[..., 1]
    width, height = outputs[..., 2], outputs[..., 3]
    x_min = (x_center - width / 2) * image_width
    x_max = (x_center + width / 2) * image_width
    y_min = (y_center - height / 2) * image_height
    y_max = (y_center + height / 2) * image_height
    
    return torch.stack([x_min, y_min, x_max, y_max], dim=-1)

# CHECK WHICH IS USEFUL TO KEEP AND REMOVE OTHERS


def process_detections(bbox_coords, obj_scores, class_probs, num_classes, max_detections=100):
    batch_size = bbox_coords.size(0)
    # Prepare lists to hold tensors for each batch item
    all_boxes = []
    all_scores = []
    all_class_probs = []

    # Thresholds
    score_threshold = 0.5
    iou_threshold = 0.5

    for i in range(batch_size):
        # Filter detections based on the objectness score
        indices = (obj_scores[i] > score_threshold).nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            # Handle case with no detections
            padded_boxes = torch.zeros((max_detections, 4), dtype=torch.float32, device=bbox_coords.device , requires_grad=True)
            padded_scores = torch.zeros(max_detections, dtype=torch.float32, device=obj_scores.device , requires_grad=True)
            padded_class_probs = torch.zeros((max_detections, num_classes), dtype=torch.float32, device=class_probs.device , requires_grad=True)
        else:
            filtered_boxes = bbox_coords[i][indices]
            filtered_scores = obj_scores[i][indices]
            filtered_class_probs = class_probs[i][indices]

            # Compute the combined score (objectness * max class probability)
            combined_scores = filtered_scores * filtered_class_probs.max(dim=1).values

            # Apply NMS to reduce boxes based on IOU threshold
            keep_indices = nms(filtered_boxes, combined_scores, iou_threshold)

            # Ensure that exactly max_detections are kept
            keep_count = keep_indices.shape[0]
            if keep_count > max_detections:
                # Keep only the top scoring detections if there are too many
                top_scores, top_indices = torch.topk(combined_scores[keep_indices], max_detections)
                keep_indices = keep_indices[top_indices]
            elif keep_count < max_detections:
                # Pad with zeros if there are too few
                pad_count = max_detections - keep_count
                keep_indices = torch.cat([
                    keep_indices,
                    torch.full((pad_count,), -1, dtype=torch.long, device=keep_indices.device)  # Use an invalid index
                ])

            # Collect final selections for this batch item
            padded_boxes = pad_tensor(filtered_boxes[keep_indices], max_detections, 0)
            padded_scores = pad_tensor(filtered_scores[keep_indices], max_detections, 0)
            padded_class_probs = pad_tensor(filtered_class_probs[keep_indices], max_detections, num_classes)
        
        all_boxes.append(padded_boxes)
        all_scores.append(padded_scores)
        all_class_probs.append(padded_class_probs)

    return {
        'boxes': torch.stack(all_boxes),
        'scores': torch.stack(all_scores),
        'class_probs': torch.stack(all_class_probs)
    }

def pad_tensor(tensor, pad_size, pad_value):
    """Pads tensor to specified size with pad_value."""
    if tensor.size(0) < pad_size:
        padding_size = pad_size - tensor.size(0)
        pad_tensor = torch.full((padding_size, *tensor.shape[1:]), pad_value, dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, pad_tensor], dim=0)
    return tensor


