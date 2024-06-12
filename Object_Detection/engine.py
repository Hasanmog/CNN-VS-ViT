import torch
import torch.nn as nn
import torch.nn.functional as F
import neptune
import json
import os
from torchvision.ops import complete_box_iou_loss
from torch.optim import Adam , lr_scheduler
from tqdm import tqdm
from postprocessing import convert_to_mins_maxes , xywh_to_xyxy , post_process_outputs , process_detections
from model import decode_outputs
from torchvision.ops import nms
from utils import  normalize_bboxes , bbox_iou , accuracy

def train(model , train_loader , val_loader ,
          epochs , lr , device , lr_schedule ,
          out_dir , 
          num_boxes_coords = 4 , 
          num_classes = 7 , # -1 for background
          num_obj_score = 1 , 
          num_anchors = 5 , 
          neptune_config = None):
    
    with open(neptune_config) as config_file:
        config = json.load(config_file)
        api_token = config.get('api_token')
        project = config.get('project')

    run = neptune.init_run(
        project=project,  
        api_token=api_token,
    )
    
    
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
    scaler = torch.cuda.amp.GradScaler()
    for epoch in tqdm(range(epochs)):
        print(f"Current Epoch : {epoch}")
        epoch_loss , loss_class , loss_obj , loss_bbox = 0 , 0 , 0 , 0
        print(f"Number of training batches ---> {len(train_loader)}")
        for idx , batch in enumerate(train_loader):
            print(f"Batch number {idx}")
            model.train(True)
            images , gt_bbox , gt_obj_scores, gt_labels , path = batch['images'] , batch['boxes'] ,batch['objectness_scores'], batch['labels'] ,batch['img_path']
            # print("images" , images.shape)
            # print("gt_obj_scores: " , gt_obj_scores, gt_obj_scores.shape , type(gt_obj_scores))
            # print("gt_bbox" , gt_bbox , gt_bbox.shape , type(gt_bbox))
            # print("gt_labels" , gt_labels , gt_labels.shape , type(gt_labels))
            gt_bbox = normalize_bboxes(gt_boxes = gt_bbox , img_height= 512 , img_width=512)
            gt_bbox = gt_bbox.to(device)
            gt_labels = gt_labels.to(device)
            gt_obj_scores = gt_obj_scores.to(device)
            images = images.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
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
                                                        num_classes=7 , max_detections=350)  
                
                
                filtered_pred_bbox = detections_filtered['boxes'].to(device)
                filtered_obj_scores = detections_filtered['scores'].to(device)
                filtered_class_probs = detections_filtered['class_probs'].to(device)
        
                class_loss = torch.nn.functional.cross_entropy(
                    filtered_class_probs.view(-1, num_classes),  # Flatten predictions to [batch_size * max_detections, num_classes]
                    gt_labels.view(-1)  # Flatten labels to [batch_size * max_detections]
                )
                class_loss = class_loss * 0.4
                loss_class += class_loss

                obj_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                filtered_obj_scores.view(-1), 
                gt_obj_scores.view(-1) 
                )  

                obj_loss = obj_loss * 0.2
                loss_obj += obj_loss
                # print("filtered_pred_bbox" ,filtered_pred_bbox.view(-1, 4)) 
                # print("gt" ,gt_bbox.view(-1, 4) )        
                bbox_loss = torch.nn.functional.smooth_l1_loss(
                    filtered_pred_bbox.view(-1, 4),  
                    gt_bbox.view(-1, 4) ####################### PROBLEM HERE #####################
                )
                bbox_loss = bbox_loss*0.4
                loss_bbox += bbox_loss
            run['batch class loss'].log(class_loss) # temporarily
            run['batch bbox loss'].log(bbox_loss) # temp
            run['batch obj loss'].log(obj_loss) #temp
    
            scaler.scale(class_loss).backward(retain_graph=True)
            scaler.scale(obj_loss).backward(retain_graph=True) 
            scaler.scale(bbox_loss).backward()
            batch_loss = class_loss + obj_loss + bbox_loss
            # batch_loss.backward()
            epoch_loss+=batch_loss
            # loss.backward()
            scaler.step(optimizer)
            scaler.update()
            if lr_schedule == "onecyclelr":
                lr_schedule.step()
                
            if idx % 200 == 0 and idx != 0:
                torch.save(model.state_dict(), os.path.join(out_dir, f'temp_checkpoint.pth'))
                model.load_state_dict(torch.load(os.path.join(out_dir, f'temp_checkpoint.pth')))
                torch.cuda.empty_cache()
                
        if lr_schedule != "oncyclelr":       
            lr_schedule.step()
        # After each epoch
        average_epoch_loss = epoch_loss / len(train_loader)
        average_loss_bbox = loss_bbox / len(train_loader)
        average_loss_class = loss_class / len(train_loader)
        average_loss_obj = loss_obj / len(train_loader)

        print(f"Average total loss for epoch {epoch} ---> {average_epoch_loss}")
        print(f"Average bbox loss for epoch {epoch} ---> {average_loss_bbox}")
        print(f"Average class loss for epoch {epoch} ---> {average_loss_class}")
        print(f"Average object score loss for epoch {epoch} ---> {average_loss_obj}")

        run['average total train loss'].log(average_epoch_loss)
        run['average class loss'].log(average_loss_class)
        run['average bbox loss'].log(average_loss_bbox)
        run['average obj loss'].log(average_loss_obj)
        
        # Validation
        model.eval()
        iou_save , acc_save , val_iou , val_acc = 0 , 0 , 0 , 0
        print(f"Number of validation batches ---> {len(val_loader)}")
        for idx , batch in enumerate(val_loader):
            print(f"Current batch number: {idx}")
            images , gt_bbox , gt_obj_scores, gt_labels , path = batch['images'] , batch['boxes'] ,batch['objectness_scores'], batch['labels'] ,batch['img_path']
            gt_bbox = normalize_bboxes(gt_boxes = gt_bbox , img_height= 512 , img_width=512)
            gt_bbox = gt_bbox.to(device)
            gt_labels = gt_labels.to(device)
            gt_obj_scores = gt_obj_scores.to(device)
            images = images.to(device)
            with torch.no_grad():
                outputs = model(images)
                batch_size , attributes , grid_size , _ = outputs.shape
                attributes_per_anchor = 4 + 1 + num_classes  
                outputs = outputs.view(batch_size, num_anchors, attributes_per_anchor, grid_size, grid_size)
                outputs = outputs.permute(0, 3, 4, 1, 2)  


                bbox_coords = outputs[..., :num_boxes_coords]  
                bbox_coords = torch.sigmoid(bbox_coords)
                pred_bbox = bbox_coords.reshape(bbox_coords.size(0), -1, bbox_coords.size(-1))
                pred_bbox = xywh_to_xyxy(pred_bbox)
                
                obj_scores = outputs[..., num_boxes_coords] 
                obj_scores = torch.sigmoid(obj_scores)
                obj_scores = obj_scores.reshape(batch_size, -1) 
                
                
                class_probs = outputs[..., num_boxes_coords + 1:]
                class_probs = F.softmax(class_probs, dim=-1)
                class_probs = class_probs.reshape(batch_size, -1, num_classes)  

                detections_filtered = process_detections(bbox_coords=pred_bbox , obj_scores= obj_scores , class_probs=class_probs , 
                                                        num_classes=7 , max_detections=350)  
                
                
                filtered_pred_bbox = detections_filtered['boxes'].to(device)
                filtered_obj_scores = detections_filtered['scores'].to(device)
                filtered_class_probs = detections_filtered['class_probs'].to(device)
            
            batch_iou = bbox_iou(filtered_pred_bbox , gt_bbox)
            batch_accuracy = accuracy(filtered_class_probs , gt_labels)
            val_iou += batch_iou
            val_acc += batch_accuracy
        
        val_average_iou = val_iou / len(val_loader)  
        val_average_acc = val_acc / len(val_loader)
        print(f"validation classification accuracy ---> {val_average_acc}")
        print(f"validation bounding box IoU ---> {val_average_iou}")
        run['validation accuracy'].log(val_average_acc)
        run['validation IoU'].log(val_average_iou)
 
        if iou_save < val_average_iou or acc_save < val_average_acc :
            checkpoint_path = os.path.join(out_dir, f'best_checkpoint.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_schedule.state_dict(),
                'train_loss': epoch_loss,
                'val_iou' : val_average_iou,
                'val_acc' : val_average_acc , 
            }, checkpoint_path)
    
'''
- Implement:
            - Validation
            - Neptune log
            - Accuracy (IoU)
            
- Clean the code
- Code for Testing
- Transform the training code processing into a function
- Add shortcuts for max_detection number between all functions
'''