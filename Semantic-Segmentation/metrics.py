import torch

def iou(outputs, targets, smooth=1e-6):
    """
    Compute the IoU (Jaccard Index) between the predicted and target masks.
    
    Args:
        outputs (torch.Tensor): the logits from your model (before sigmoid/softmax).
        targets (torch.Tensor): the ground truth labels.
        smooth (float): A small value to avoid division by zero.

    Returns:
        float: the computed IoU.
    """
    # Assuming outputs are probabilities (after sigmoid/softmax)
    outputs = torch.sigmoid(outputs) > 0.5  # Threshold probabilities to get binary tensor
    outputs = outputs.float()  # Convert boolean tensor to float

    # Flatten the tensors to simplify computation
    outputs = outputs.view(-1)
    targets = targets.view(-1)
    
    # Compute Intersection and Union
    intersection = (outputs * targets).sum()
    total = (outputs + targets).sum()
    union = total - intersection

    # Compute the IoU
    IoU = (intersection + smooth) / (union + smooth)
    return IoU

