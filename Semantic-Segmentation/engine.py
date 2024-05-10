import neptune
import json
import torch
from torch.optim import Adam , lr_scheduler
from torch.nn import BCELoss
from tqdm import tqdm
from utils import calculate_iou


def train_val(model ,train_loader , val_loader, epochs , lr , lr_schedule , out_dir , device , neptune_config):

    with open(neptune_config) as config_file:
            config = json.load(config_file)
            api_token = config.get('api_token')
            project = config.get('project')
        
    run = neptune.init_run(
                    project=project,  # specify your project name here
                    api_token= api_token,
                    #with_id = 'VLMEO-1048'
    )   
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr , weight_decay=1e-4)
    loss_function = BCELoss()
    
    if lr_schedule == "onecyclelr":
        lr_schedule = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs, pct_start=0.2)
        
    elif lr_schedule == "multi_step_lr":
        lr_drop_list = [5, 10 , 15 , 20 ,25 ,30 ,35, 40]
        lr_schedule = lr_scheduler.MultiStepLR(optimizer, milestones=lr_drop_list , gamma = 0.2)
        
    elif lr_schedule == "step_lr":
        step_size = 10
        gamma = 0.5
        lr_schedule = lr_scheduler.StepLR(optimizer, step_size = step_size , gamma = gamma)
    else:
        gamma = 0.97
        lr_schedule = lr_scheduler.ExponentialLR(optimizer , gamma)
    
    start_epoch = 0
    best = float('inf')
    for epoch in tqdm(range(start_epoch , epochs) , desc = "training"):    
        print(f"Epoch : {epoch}")
        model.train(True)
        batch_loss = 0.0
        ious = 0.0
        train_iou = 0.0
        train_loss = 0.0
        
        for i , batch in enumerate(train_loader):
            images , masks = batch
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            # print("outputs" , outputs)
            # print("masks" , masks)
            
            
            loss = loss_function(outputs, masks)
            iou = calculate_iou(preds=outputs , labels = masks)
            batch_loss += loss.item()
            ious += iou
            loss.backward()
            optimizer.step()
            
        lr_schedule.step()
        train_loss = batch_loss / len(train_loader)
        train_iou = ious / len(train_loader)
        run["train loss"].log(train_loss)
        run["train_iou"].log(train_iou)
        current_lr = optimizer.param_groups[0]['lr']
        run['lr'].log(current_lr)
        print(f"train_loss-------------------------------->{train_loss}")
        print(f"train_IoU-------------------------------->{train_iou}")
        
        
        model.eval()
        ious = 0.0
        validation_loss = 0.0
        val_loss =0.0
        val_iou = 0.0
        for i , batch in enumerate(val_loader):
            images , masks = batch
            images = images.to(device)
            masks = masks.to(device)
            with torch.no_grad():
                outputs = model(images)
                loss = loss_function(outputs, masks)
                iou = calculate_iou(preds=outputs , labels = masks)
                validation_loss += loss.item()
                ious += iou
            
        val_loss = validation_loss / len(val_loader)
        val_iou = ious / len(val_loader)
        run["validation loss"].log(val_loss)
        run["validation IoU"].log(val_iou)
        print(f"val_loss-------------------------------->{val_loss}")
        print(f"val_IoU-------------------------------->{val_iou}")
        
        if val_loss < best :  
            torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_schedule.state_dict(),
                    'train_loss': train_loss,
                    'train_iou' : train_iou , 
                    'val_loss': val_loss , 
                    'val_iou' : val_iou ,
                }, out_dir )  
            best = val_loss
           
    return train_loss , train_iou , val_loss , val_iou 


# def test():
    