import neptune
import json
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam , lr_scheduler
from torch.nn.functional import binary_cross_entropy_with_logits
from tqdm import tqdm
from utils import iou , calc_f1_score ,dice_loss , matching_algorithm
from torch.cuda.amp import GradScaler, autocast


def train_val(model ,train_loader , val_loader, epochs , lr , lr_schedule , out_dir ,device , neptune_config , resume_checkpoint = None):

    with open(neptune_config) as config_file:
            config = json.load(config_file)
            api_token = config.get('api_token')
            project = config.get('project')
        
    run = neptune.init_run(
                    project=project,  # specify your project name here
                    api_token= api_token,
                    #with_id = 'SOL-91'
    )
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr , weight_decay=1e-4)
    if resume_checkpoint:   
        checkpoint = torch.load(resume_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # lr_schedule.load_state_dict(checkpoint['lr_schedule_state_dict'])
        epochs = checkpoint['epoch']
        for g in optimizer.param_groups:
            g['lr'] = lr
        print("Resuming from checkpoint")
    
    
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
        
    scaler = GradScaler()

    start_epoch = 0
    best = float('inf')
    for epoch in tqdm(range(start_epoch , epochs) , desc = "training"):    
        print(f"Epoch : {epoch}")
        model.train(True)
        batch_loss = 0.0
        ious_3 = 0.0
        ious_5 = 0.0
        ious_7 = 0.0
        train_iou = 0.0
        train_loss = 0.0
        
        for i , batch in enumerate(train_loader):
            images , masks = batch
            images = images.to(device)
            masks = masks.float()
            masks = masks.to(device)
            optimizer.zero_grad()  # Reset gradients to zero for new mini-batch

            outputs = model(images)  # Forward pass
            loss = binary_cross_entropy_with_logits(outputs, masks)  # Compute loss
            
            iou_3 , iou_5 , iou_7 = iou(preds=outputs, labels=masks)
            batch_loss += loss.item()
            ious_3 += iou_3
            ious_5 += iou_5
            ious_7 += iou_7
            
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            
        lr_schedule.step()
        train_loss = batch_loss / len(train_loader)
        train_iou_3 = ious_3 / len(train_loader)
        train_iou_5 = ious_5 / len(train_loader)    
        train_iou_7 = ious_7 / len(train_loader)
        run["train loss"].log(train_loss)
        run["train IoU@0.3"].log(train_iou_3)
        run["train IoU@0.5"].log(train_iou_5)
        run["train IoU@0.75"].log(train_iou_7)
        current_lr = optimizer.param_groups[0]['lr']
        run['lr'].log(current_lr)
        print(f"train_loss-------------------------------->{train_loss}")
        print(f"train_IoU @0.3 -------------------------------->{train_iou_3}")
        print(f"train_IoU @0.5 -------------------------------->{train_iou_5}")
        print(f"train_IoU @0.75 -------------------------------->{train_iou_7}")
        
        
        model.eval()
        ious = 0.0
        validation_loss = 0.0
        val_loss =0.0
        val_iou_3 = 0.0
        val_iou_5 = 0.0
        val_iou_7 = 0.0
        for i , batch in enumerate(val_loader):
            images , masks = batch
            images = images.to(device)
            masks = masks.to(device)
            with torch.no_grad(): #, autocast():
                outputs = model(images)
                loss = binary_cross_entropy_with_logits(outputs, masks)
                iou_3 , iou_5 , iou_7 = iou(preds=outputs , labels = masks)
                validation_loss += loss.item()
                val_iou_3 += iou_3
                val_iou_5 += iou_5
                val_iou_7 += iou_7
            
        val_loss = validation_loss / len(val_loader)
        val_3 = val_iou_3/ len(val_loader)
        val_5 = val_iou_5/ len(val_loader)
        val_7 = val_iou_7/ len(val_loader)
        run["validation loss"].log(val_loss)
        run["Val IoU@0.3"].log(val_3)
        run["Val IoU@0.5"].log(val_5)
        run["Val IoU@0.75"].log(val_7)
        print(f"val_loss-------------------------------->{val_loss}")
        print(f"val_IoU @0.3 -------------------------------->{val_3}")
        print(f"val_IoU @0.5 -------------------------------->{val_5}")
        print(f"val_IoU @0.75 -------------------------------->{val_7}")
        
        if val_loss < best :  
            torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_schedule.state_dict(),
                    'train_loss': train_loss,
                    'train_iou' : train_iou , 
                    'val_loss': val_loss , 
                }, out_dir )  
            best = val_loss
           
    return train_loss , val_loss


def test(model , test_loader , checkpoint:str , device , output_dir:str):
    
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()
    
    test_loss = 0.0
    test_ious_3 = 0.0
    test_ious_5 = 0.0
    test_ious_7 = 0.0
    fs = 0.0
    precisions = 0.0
    recalls = 0.0
    dices = 0.0
    for i , batch in tqdm(enumerate(test_loader) , desc = "Testing progress"):
        images , masks = batch
        images = images.to(device)
        masks = masks.to(device)
        
        
        with torch.no_grad():
            outputs = model(images)
            loss = binary_cross_entropy_with_logits(outputs, masks)
            iou_3 , iou_5 , iou_7 = iou(preds=outputs , labels = masks)
            test_ious_3 += iou_3
            test_ious_5 += iou_5
            test_ious_7 += iou_7
            test_loss += loss.item()
            precision, recall, f1 = calc_f1_score(predicted_masks=outputs , gt_masks=masks)
            dice = dice_loss(predicted_masks=outputs , gt_masks=masks)
            dices += dice.item()
            fs += f1
            precisions += precision
            recalls += recall
            
            
            
    final_loss =  test_loss/len(test_loader)
    final_iou_3 = test_ious_3/len(test_loader)
    final_iou_5 = test_ious_5 / len(test_loader)
    final_iou_7 = test_ious_7 / len(test_loader)
    final_precision = precisions/len(test_loader)
    final_recall = recalls/len(test_loader)
    final_f1 = fs/len(test_loader)
    final_dice = dices/len(test_loader)      
    print(f"Test Loss -------------> {final_loss}")
    print(f"Test IoU@0.3 -------------> {final_iou_3}")
    print(f"Test IoU@0.5 -------------> {final_iou_5}")
    print(f"Test IoU@0.75 -------------> {final_iou_7}")
    print(f"Test Precision Score -------------> {final_precision}")
    print(f"Test Recall Score -------------> {final_recall}")
    print(f"Test F1 Score -------------> {final_f1}")
    print(f"Test Dice Loss -------------> {final_dice}")
    
    results = {
        "Test Loss": final_loss,
        "Test IoU@0.3": final_iou_3,
        "Test IoU@0.5": final_iou_5,
        "Test IoU@0.75": final_iou_7,
        "Test Precision": final_iou_5,
        "Test Recall": final_iou_7,
        "Test F1": final_f1,
        "Test Dice": final_dice
    }

    out = os.path.join(output_dir, "results.json")
    
    with open(out, 'w') as f:
        for key, value in results.items():
            json.dump({key: value}, f)
            f.write('\n')
    
    return results
