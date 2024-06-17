import torch
import torch.nn as nn
import torch.nn.functional as F
import neptune
import json
import os
from torch.cuda.amp import GradScaler, autocast
from torchvision.ops import sigmoid_focal_loss
from torch.optim import Adam , lr_scheduler
from tqdm import tqdm
from postprocessing import convert_to_mins_maxes , xywh_to_xyxy 
from torchvision.ops import nms
from utils import calculate_accuracy , calculate_iou , calculate_precision_recall

def train(model , train_loader , val_loader ,
          epochs , lr , device , lr_schedule ,
          out_dir , 
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
    scaler = GradScaler()
    # Initialize the learning rate scheduler based on user input
    if lr_schedule == "onecyclelr":
        lr_schedule = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs, pct_start=0.2)
    elif lr_schedule == "multi_step_lr":
        lr_schedule = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20, 25, 30, 35, 40], gamma=0.2)
    elif lr_schedule == "step_lr":
        lr_schedule = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        lr_schedule = lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
        
    save = 0
    for epoch in tqdm(range(epochs)):
        print(f"Current Epoch : {epoch}")
        epoch_loss , loss_class , loss_reg , loss_center = 0 , 0 , 0 , 0 
        print(f"Number of training batches ---> {len(train_loader)}")
        for idx , batch in enumerate(train_loader):
            print(f"Batch number {idx}")
            batch_loss = 0
            model.train(True)
            img , regression , cls , centerness = batch
            img = img.to(device)
            regression = regression.to(device)
            cls = cls.to(device)
            centerness = centerness.to(device)
            optimizer.zero_grad()
            
            with autocast():
                cls_pred , reg_pred , center_pred = model(img)
                
                regression_loss = F.smooth_l1_loss(reg_pred , regression)
                loss_reg += regression_loss.item()
                scaler.scale(regression_loss).backward(retain_graph=True)
                run['regression_loss'].log(regression_loss)
                
                center_loss = F.binary_cross_entropy_with_logits(center_pred , centerness)
                loss_center += center_loss.item()
                scaler.scale(center_loss).backward(retain_graph=True)
                run['center_loss'].log(center_loss)
                
                cls_loss = sigmoid_focal_loss(cls_pred , cls , alpha=0.25, gamma=2.0, reduction='mean') 
                loss_class += cls_loss.item()
                scaler.scale(cls_loss).backward()
                run['cls_loss'].log(cls_loss)
                batch_loss = cls_loss + center_loss + regression_loss
                epoch_loss += batch_loss
                print(f"Batch Loss ---> {batch_loss}")
                print(f"Batch cls Loss ---> {loss_class}")
                print(f"Batch center Loss ---> {loss_center}")
                print(f"Batch regression Loss ---> {loss_reg}")
                
            
                scaler.step(optimizer)
                scaler.update()
            if lr_schedule == 'onecyclelr':
                lr_schedule.step()
        if lr_schedule != 'onecyclelr':
            lr_schedule.step()
        run['lr'].log(lr_schedule.get_last_lr()[0])
        print(f"Epoch Loss ---> {epoch_loss/len(train_loader)}")
        run['Epoch Loss'].log(epoch_loss)
        total_cls_loss = loss_class/len(train_loader)
        total_loss_center = loss_center/len(train_loader)
        total_reg_loss = loss_reg/len(train_loader)
        print(f"Epoch cls Loss ---> {total_cls_loss}")
        print(f"Epoch center Loss ---> {total_loss_center}")
        print(f"Epoch regression Loss ---> {total_reg_loss}")
        
        print("-----------------------")
        print(f"Validation for epoch {epoch}")
        model.eval()
        ious = 0
        for idx , batch in enumerate(val_loader):
            img , regression , cls , centerness = batch
            img = img.to(device)
            regression = regression.to(device)
            cls = cls.to(device)
            centerness = centerness.to(device)
            
            with torch.no_grad():
                
                cls_pred , reg_pred , center_pred = model(img)
                
            # Calculate metrics
                accuracy = calculate_accuracy(cls_pred, cls)
                iou = calculate_iou(reg_pred , regression , 0.5)
                ious+=iou
                precision_05, recall_05 = calculate_precision_recall(reg_pred, regression, 0.5)
                precision_07, recall_07 = calculate_precision_recall(reg_pred, regression, 0.7)
                precision_09, recall_09 = calculate_precision_recall(reg_pred, regression, 0.9)
                
                # Log metrics
                run['accuracy'].log(accuracy)
                run['IoU'].log(iou)
                run['precision@0.5'].log(precision_05)
                run['recall@0.5'].log(recall_05)
                run['precision@0.7'].log(precision_07)
                run['recall@0.7'].log(recall_07)
                run['precision@0.9'].log(precision_09)
                run['recall@0.9'].log(recall_09)
                
        avg_iou = ious/len(val_loader)
        if avg_iou > save :
            checkpoint_path = os.path.join(out_dir, f'best_checkpoint.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_schedule.state_dict(),
                'train_loss': epoch_loss,
                'val_acc': accuracy,
                'val_IoU' : iou
            }, checkpoint_path)
            save = avg_iou
        log_json = {
            'epoch' : epoch , 
            'lr' : lr_schedule.get_last_lr()[0] , 
            'train cls loss' : total_cls_loss , 
            'train center loss' : total_loss_center , 
            'train regression loss' : total_reg_loss , 
            'Train total loss' : epoch_loss , 
            'Val Accuracy' : accuracy , 
            'Val IoU' : iou    
        }        
        log_file_path = os.path.join(out_dir, f'log.json')
        with open(log_file_path, 'a') as f:
            json.dump(log_json , f)
            f.write('\n')
        print(f"Validation for epoch {epoch} completed.")
    run.stop()
                
                
                