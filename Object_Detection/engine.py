import torch
import torch.nn.functional as F
import neptune
import json
from torch.optim import Adam
from torchvision.ops import box_iou
from torch.cuda.amp import GradScaler, autocast

# Taken from https://github.com/eriklindernoren/PyTorch-YOLOv3
def compute_loss(pred_bbox, pred_labels, pred_obj_scores, gt_bbox, gt_labels, num_classes):
    device = pred_bbox.device  # Assuming all tensors are on the same device
    
    # Constants for losses weights
    lambda_coord = 5.0  # Weight for bbox loss
    lambda_noobj = 0.5  # Weight for no object loss
    lambda_obj = 1.0    # Weight for object loss
    lambda_class = 1.0  # Weight for classification loss
    
    # Calculate IoU between predicted and ground truth boxes
    iou = box_iou(pred_bbox, gt_bbox)
    
    # Objectness Loss: We calculate this as Binary Cross-Entropy
    obj_loss = F.binary_cross_entropy(pred_obj_scores, iou.max(dim=1).values.unsqueeze(1), reduction='none')
    
    # Bounding Box Loss (using IoU loss as a proxy here, though typically you'd use CIoU or DIoU etc.)
    bbox_loss = F.mse_loss(pred_bbox, gt_bbox, reduction='none').sum(dim=1)  # Sum loss over coordinates
    
    # Classification Loss (assuming pred_labels and gt_labels are already in the appropriate format)
    class_loss = F.cross_entropy(pred_labels, gt_labels, reduction='none')
    
    # Only consider bbox loss and class loss where there's an object (iou.max(dim=1).values > some threshold)
    object_mask = iou.max(dim=1).values > 0.5  # Threshold for determining an object presence
    bbox_loss = bbox_loss * object_mask.float()
    class_loss = class_loss * object_mask.float()
    
    # Weighted sum of losses
    total_loss = (lambda_coord * bbox_loss.mean() + 
                  lambda_obj * obj_loss[object_mask].mean() +
                  lambda_noobj * obj_loss[~object_mask].mean() +
                  lambda_class * class_loss.mean())
    
    return total_loss
    


def train(model , train_loader , val_loader ,lr ,lr_scheduler, epochs , out_dir , device , neptune_config):
    
    with open(neptune_config) as config_file:
        config = json.load(config_file)
        api_token = config.get('api_token')
        project = config.get('project')

    run = neptune.init_run(
        project=project, 
        api_token=api_token,
        # with_id='SOL-91'  # Uncomment if you need to specify a particular run ID
    )
    
    model.to(device)
    optimizer = Adam(model.parameters() , lr = lr)
    if lr_schedule == "onecyclelr":
        lr_schedule = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs, pct_start=0.2)
    elif lr_schedule == "multi_step_lr":
        lr_schedule = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20, 25, 30, 35, 40], gamma=0.2)
    elif lr_schedule == "step_lr":
        lr_schedule = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        lr_schedule = lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
        
    scaler = GradScaler()
    
    for idx , sample in enumerate(train_loader):
        
        img , gt_bbox , gt_label = sample['image_tensor'] , sample['bboxes'] , sample['category']
        
        for epoch in range(epochs):
            print(f"current epoch :{epoch}")
    
    