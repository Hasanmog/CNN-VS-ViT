import torch
import torch.nn.functional as F
import neptune
import json
from model import decode_outputs
from postprocessing import convert_to_mins_maxes , non_max_suppression , process_boxes
from torch.optim import Adam , lr_scheduler
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
    


def train(model , train_loader , val_loader ,lr ,lr_schedule, epochs , out_dir , device , neptune_config = None):
    
    if neptune_config != None :
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
    for epoch in range(epochs):
        print(f"current epoch :{epoch}")
        for idx , sample in enumerate(train_loader):
            img , gt_bbox , gt_label = sample['image_tensor'] , sample['bboxes'] , sample['category']
            model.train(True)
            with autocast():
                outputs = model(img)
                boxes , object , class_scores = decode_outputs(outputs) 
                boxes = boxes.reshape(-1, 4)
                class_scores = class_scores.reshape(-1, 6)
                assert boxes.shape[0] == class_scores.shape[0], "Mismatch in bounding boxes and class scores counts"
                picked_boxes, picked_scores, picked_classes = non_max_suppression(boxes,class_scores)
            
            loss = compute_loss(pred_bbox=picked_boxes , pred_labels=picked_classes , pred_obj_scores= picked_scores,
                                gt_bbox= gt_bbox , gt_labels = gt_label , num_classes=6)
            
            print(f"Train loss:{loss}")
        model.eval()
        for idx , sample in enumerate(val_loader):
            img , gt_bbox , gt_label = sample['image_tensor'] , sample['bboxes'] , sample['category']
            with torch.no_grad(), autocast():
                outputs = model(img)
                boxes , object , class_scores = decode_outputs(outputs) 
                boxes = boxes.reshape(-1, 4)
                class_scores = class_scores.reshape(-1, 6)
                assert boxes.shape[0] == class_scores.shape[0], "Mismatch in bounding boxes and class scores counts"
                picked_boxes, picked_scores, picked_classes = non_max_suppression(boxes,class_scores)
            
            loss = compute_loss(pred_bbox=picked_boxes , pred_labels=picked_classes , pred_obj_scores= picked_scores,
                                gt_bbox= gt_bbox , gt_labels = gt_label , num_classes=6)
            
            print(f"val loss:{loss}")
                        
    