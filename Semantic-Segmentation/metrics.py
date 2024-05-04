import torch

def iou(outputs, labels, smooth=1e-6):
    """
    Compute the IoU (Intersection over Union) metric.
    
    Parameters:
    outputs (torch.Tensor): The model predictions, shape (N, C, H, W),
                            where N is the batch size, C is the number of classes,
                            H and W are the dimensions of the image.
    labels (torch.Tensor): The ground truth labels, shape (N, H, W) with class indices.
    
    Returns:
    float: The average IoU score over the batch.
    """
    # Assume outputs are raw logits from the network
    outputs = torch.sigmoid(outputs) > 0.5  # Apply sigmoid and threshold to get binary map
    outputs = outputs.float()  # Convert boolean tensor to float
    
    # Create a one-hot encoding of labels (N, C, H, W)
    labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=outputs.shape[1]).permute(0, 3, 1, 2).float()
    
    # Calculate intersection and union
    intersection = (outputs * labels_one_hot).sum(dim=(2, 3))  # Sum over height and width
    union = (outputs + labels_one_hot - outputs * labels_one_hot).sum(dim=(2, 3))  # Sum over height and width
    
    # Compute IoU and average over batch
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()
