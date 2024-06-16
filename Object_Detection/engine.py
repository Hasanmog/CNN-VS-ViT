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
        print(f"Epoch cls Loss ---> {loss_class/len(train_loader)}")
        print(f"Epoch center Loss ---> {loss_center/len(train_loader)}")
        print(f"Epoch regression Loss ---> {loss_reg/len(train_loader)}")
                
                
                
                