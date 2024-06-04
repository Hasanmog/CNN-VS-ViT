
import numpy as np
import torch
import cv2


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
    Convert XYWH format (x,y center point and width, height) to XYXY format (x,y top left and x,y bottom right).
    :param xywh: [X, Y, W, H]
    :return: [X1, Y1, X2, Y2]
    """
    # xywh = xywh.cpu()
    # xywh = xywh.detach().numpy()
    # print(xywh)
    if np.array(xywh).ndim > 1 or len(xywh) > 4:
        raise ValueError('xywh format: [x1, y1, width, height]')
    x1 = xywh[0] - xywh[2] / 2
    y1 = xywh[1] - xywh[3] / 2
    x2 = xywh[0] + xywh[2] / 2
    y2 = xywh[1] + xywh[3] / 2
    return np.array([int(x1), int(y1), int(x2), int(y2)])


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