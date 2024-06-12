
import numpy as np
import cv2
import torch
from torchvision.ops import nms
import torchvision.ops as ops
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




