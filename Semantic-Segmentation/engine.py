import torch
import torch.nn as nn
import neptune
from torch.nn import functional
from tqdm import tqdm
from metrics import iou

def train_one_epoch(model , train_loader , val_loader ,epochs , lr , scheduler , out_dir , device):
    
    run = neptune.init_run(
    project='Solo/Paper2code',  # specify your project name here
    api_token= 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzZjFkNjIxNy1jOWZlLTQzZTUtYTA3OS0zMjhlM2UwNTMzODYifQ==',
    #with_id = 'VLMEO-1048'
    )   
    
    batch_loss = 0
    batch_iou = 0
    val_meter = float('inf')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters() , lr = lr)
    
    if scheduler == "onecyclelr":
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs, pct_start=0.2)
    elif scheduler == "multi_step_lr":
        lr_drop_list = [4, 8]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_drop_list)
    elif scheduler == "step_lr":
        step_size = 10
        gamma = 0.5
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = step_size , gamma = gamma)
    else:
        gamma = 0.95
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer , gamma)
        
    
    
    
    start_epoch = 0
    accum_steps = 4
    for epoch in tqdm(range(start_epoch, epochs), desc="Epoch Progress", leave=True):
        batch_loss = 0.0  # Reset loss counter for the training batches
        batch_iou = 0.0   # Reset IoU counter for the training batches
        print("Current Epoch:", epoch + 1)
        model.train(True)
        
        acc_counter = 0  # Reset accumulation counter at the start of each epoch
        optimizer.zero_grad()  # Zero gradients at the start of each epoch, not each batch
        
        for i, batch in tqdm(enumerate(train_loader), desc="Training Batching Progress", leave=False):
            images, masks = batch
            images = images.to(device)
            masks = masks.to(device)
            
            pred = model(images)
            loss = functional.binary_cross_entropy(pred, masks)
            IOU = iou(pred, masks)
            loss = loss / accum_steps  # Scale loss
            loss.backward()  # Accumulate gradients
            
            batch_loss += loss.item() * accum_steps  # Correctly scale back up the loss for reporting
            batch_iou += IOU.item()

            acc_counter += 1  # Increment accumulation counter
            if acc_counter % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()  # Zero gradients after updating
            
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        run['Learning Rate'].log(current_lr)
        train_loss = batch_loss / len(train_loader)
        train_iou = batch_iou / len(train_loader)
        print("Current LR:", current_lr)
        print("Final Training Loss:", train_loss)
        print("Final Training IoU:", train_iou)

        
        
        model.eval()
        batch_loss = 0.0
        batch_iou = 0.0
        
        with torch.no_grad():
            for i , batch in enumerate(val_loader):
                
                images , masks = batch
                
                images = images.to(device)
                masks = masks.to(device)
                pred = model(images)
                loss = functional.binary_cross_entropy(pred , masks)
                IoU = iou(pred , masks)
                batch_loss += loss.item()
                batch_iou += IoU.item()
            
        val_loss = batch_loss / len(val_loader)
        val_iou = batch_iou / len(val_loader)
        print("Final Validation Loss:", val_loss)
        print("Final Validation IoU:", val_iou)
        
        if val_loss < val_meter:
            
            torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss , 
                    'train_iou' : train_iou , 
                    'val_iou' : val_iou
                }, out_dir )
            
            val_meter = val_loss
        
        run['training loss per Epoch'].log(train_loss)
        run['Training IoU per Epoch'].log(train_iou)
        run['validation loss per Epoch'].log(val_loss)
        run['Validation IoU per Epoch'].log(val_iou)
        
    metrics = {
            'train_loss' : train_loss , 
            'val_loss' : val_loss ,
            'train_iou' : train_iou , 
            'val_iou' : val_iou
        }
        
    run.stop()
    return metrics
    

"""
TODO: 

[] Error free code  
                    - non logical IoU results
                    - very Large Model (1.2GB)
[] Add test_one_step function for testing
[] Visualize different feature maps at different layers in encoder and decoder (Create a generalized function for that)

"""