import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import complete_box_iou_loss
from torch.optim import Adam , lr_scheduler
from tqdm import tqdm
from postprocessing import convert_to_mins_maxes , xywh_to_xyxy , post_process_outputs
from model import decode_outputs
from torchvision.ops import nms
from utils import ciou



def loss(gt_bbox , pred_bbox):
    
    box_iou_loss = complete_box_iou_loss(gt_bbox , pred_bbox)
    

def train(model , train_loader , val_loader ,
          epochs , lr , device , lr_schedule ,
          num_boxes_coords = 4 , 
          num_classes = 6 ,
          num_obj_score = 1 , 
          num_anchors = 5 , 
          neptune_config = None):
    
    model = model.to(device)
    optimizer = Adam(model.parameters() , lr = lr)
    
    
    # Initialize the learning rate scheduler based on user input
    if lr_schedule == "onecyclelr":
        lr_schedule = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs, pct_start=0.2)
    elif lr_schedule == "multi_step_lr":
        lr_schedule = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20, 25, 30, 35, 40], gamma=0.2)
    elif lr_schedule == "step_lr":
        lr_schedule = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        lr_schedule = lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    
    attributes_per_anchor = num_boxes_coords+ num_obj_score + num_classes  
        
    for epoch in tqdm(range(epochs)):
        model.train(True)
        print(f"Current Epoch : {epoch}")
        for idx , batch in enumerate(train_loader):

            images , gt_bbox , gt_labels , _ = batch['images'] , batch['boxes'] , batch['labels'] , batch['img_path']   
            images = images.to(device)
            outputs = model(images)
            batch_size , attributes , grid_size , _ = outputs.shape
            outputs = outputs.view(batch_size, num_anchors, attributes_per_anchor, grid_size, grid_size) # 4 , 5 , 11 , 61 , 61
            outputs = outputs.permute(0, 3, 4, 1, 2) # 4 , 61 , 61 , 5 , 11
            bbox_coords = outputs[..., :num_boxes_coords]  # each cell in the 61 x 61 grid has five boxes each of 4 coords (x , y , h , w)
            obj_scores = outputs[..., num_boxes_coords] 
            obj_scores = torch.sigmoid(obj_scores)# each of these boxes have an object score (presence or absence)
            class_probs = outputs[..., num_boxes_coords + 1:]
            class_probs = F.softmax(class_probs, dim=-1)# also each box have a class probability(6 probabilities) indicating which class is present in this box
            bboxes_corners = convert_to_mins_maxes(bbox_coords)

            # TO BE CONTINUED AFTER CHECKING THE SHAPE OF EACH VARIABLE (add print statements after each line of code)
    
                        
    