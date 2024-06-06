import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import complete_box_iou_loss
from torch.optim import Adam , lr_scheduler
from tqdm import tqdm
from postprocessing import convert_to_mins_maxes , xywh_to_xyxy , post_process_outputs , process_detections
from model import decode_outputs
from torchvision.ops import nms
from utils import  normalize_bboxes , bbox_iou

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def train(model , train_loader , val_loader ,
          epochs , lr , device , lr_schedule ,
          num_boxes_coords = 4 , 
          num_classes = 7 , # -1 for background
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
            
            
            images , gt_bbox , gt_obj_scores, gt_labels , path = batch['images'] , batch['boxes'] ,batch['objectness_scores'], batch['labels'] ,batch['img_path']
            # print("images" , images.shape)
            # print("gt_obj_scores: " , gt_obj_scores, gt_obj_scores.shape , type(gt_obj_scores))
            # print("gt_bbox" , gt_bbox , gt_bbox.shape , type(gt_bbox))
            # print("gt_labels" , gt_labels , gt_labels.shape , type(gt_labels))
            gt_bbox = normalize_bboxes(gt_boxes = gt_bbox , img_height= 512 , img_width=512)
            print("gt_bbox" , gt_bbox.shape , type(gt_bbox))
            gt_labels = gt_labels.to(device)
            gt_obj_scores = gt_obj_scores.to(device)
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            
            batch_size , attributes , grid_size , _ = outputs.shape
            attributes_per_anchor = 4 + 1 + num_classes  # Bbox coords (4) + objectness score (1) + class probabilities
            outputs = outputs.view(batch_size, num_anchors, attributes_per_anchor, grid_size, grid_size)
            outputs = outputs.permute(0, 3, 4, 1, 2)  # Reorder for processing: [batch, grid_size, grid_size, num_anchors, attributes_per_anchor]


            bbox_coords = outputs[..., :num_boxes_coords]  # each cell in the 61 x 61 grid has five boxes each of 4 coords (x , y , h , w)
            bbox_coords = torch.sigmoid(bbox_coords)
            pred_bbox = bbox_coords.reshape(bbox_coords.size(0), -1, bbox_coords.size(-1))
            pred_bbox = xywh_to_xyxy(pred_bbox)
            # print("sample bbox" , nested_lists[0:2][:5]
            obj_scores = outputs[..., num_boxes_coords] 
            obj_scores = torch.sigmoid(obj_scores)# each of these boxes have an object score (presence or absence)
            obj_scores = obj_scores.reshape(batch_size, -1)  # Flatten grid and anchor dimensions
            # print("output obj scores: " , obj_scores.shape)
            
            class_probs = outputs[..., num_boxes_coords + 1:]
            class_probs = F.softmax(class_probs, dim=-1)# also each box have a class probability(6 probabilities) indicating which class is present in this box
            class_probs = class_probs.reshape(batch_size, -1, num_classes)  # Flatten grid and anchor, keep classes separate
            # print("class_probs" , class_probs.shape)
            # TO BE CONTINUED AFTER CHECKING THE SHAPE OF EACH VARIABLE (add print statements after each line of code)

            detections_filtered = process_detections(bbox_coords=pred_bbox , obj_scores= obj_scores , class_probs=class_probs , 
                                                     num_classes=7 , max_detections=100)  
            
            
            filtered_pred_bbox = detections_filtered['boxes'].to(device)
            filtered_obj_scores = detections_filtered['scores'].to(device)
            filtered_class_probs = detections_filtered['class_probs'].to(device)  
            print("boxes" , filtered_pred_bbox.shape)
            print("class scores" , filtered_class_probs.shape)
            # Assuming gt_labels shape is [4, 100] (batch_size, max_detections)
            # Ensure gt_labels are within the expected range for class indices (0 to num_classes-1)

            # Adjust gt_labels to match shape for cross entropy
            # filtered_class_probs is [batch_size, max_detections, num_classes]
            print("Filtered Class Probabilities Shape:", filtered_class_probs.shape)
            print("Ground Truth Labels Shape:", gt_labels.shape)
            print("Unique values in gt_labels:", torch.unique(gt_labels))
            class_loss = torch.nn.functional.cross_entropy(
                filtered_class_probs.view(-1, num_classes),  # Flatten predictions to [batch_size * max_detections, num_classes]
                gt_labels.view(-1)  # Flatten labels to [batch_size * max_detections]
            )
            print("class_loss" , class_loss)
            class_loss = class_loss * 0.4
            class_loss.backward(retain_graph=True)
            
            # print("obj scores" , filtered_obj_scores.shape)
            # print("after:" , filtered_obj_scores.view(-1).shape)
            # print("gt_obj_scores" ,type(gt_obj_scores) ,  gt_obj_scores.shape)
            # print("after:" , gt_obj_scores.view(-1).shape)
            # if gt_obj_scores.dtype != torch.float32:
            #     gt_obj_scores = gt_obj_scores.float()

            obj_loss = torch.nn.functional.binary_cross_entropy(
            filtered_obj_scores.view(-1), 
            gt_obj_scores.view(-1) 
            )  
            print("obj loss" , obj_loss)
            obj_loss = obj_loss * 0.2
            obj_loss.backward(retain_graph=True)          
            
            bbox_loss = torch.nn.functional.smooth_l1_loss(
                filtered_pred_bbox.view(-1, 4),  
                gt_bbox.view(-1, 4)
            )
            bbox_loss = bbox_loss*0.4
            bbox_loss.backward()
            
            # loss.backward()
            optimizer.step()