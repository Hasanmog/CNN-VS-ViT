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
from utils import calculate_accuracy , calculate_iou , calculate_precision_recall , extract_bounding_boxes , regression_map_to_boxes

def train(model , train_loader , val_loader ,
          epochs , lr , device , lr_schedule ,
          out_dir ,   
          accum_steps = 4 , 
          neptune_config = None):
    
    with open(neptune_config) as config_file:
        config = json.load(config_file)
        api_token = config.get('api_token')
        project = config.get('project')

    model = model.to(device)
    optimizer = Adam(model.parameters() , lr = lr , weight_decay=1e-5 , amsgrad=False)
    scaler = GradScaler()
    if lr_schedule == "onecyclelr":
        lr_schedule = lr_scheduler.OneCycleLR(optimizer, max_lr=1e-6, steps_per_epoch=len(train_loader), epochs=epochs, pct_start=0.2)
    elif lr_schedule == "multi_step_lr":
        lr_schedule = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20, 25, 30, 35, 40], gamma=0.2)
    elif lr_schedule == "step_lr":
        lr_schedule = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif lr_schedule == "cosineanneal":
        lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    else:
        lr_schedule = lr_scheduler.ExponentialLR(optimizer, gamma=0.994)
    
    run = neptune.init_run(
        project=project,  
        api_token=api_token,
    )  
    saver , saver_acc = 0 , 0
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
            wclass = 0.4
            wregress = 0.4
            wcenter = 0.2
            with autocast():
                cls_pred , reg_pred , center_pred = model(img)
                pred_boxes , gt_boxes = extract_bounding_boxes(reg_pred , regression)
                regression_loss = F.smooth_l1_loss(pred_boxes , gt_boxes)
                loss_reg += regression_loss.item()
                run['regression_loss'].log(regression_loss)
                
                center_loss = F.binary_cross_entropy_with_logits(center_pred , centerness)
                loss_center += center_loss.item()
                run['center_loss'].log(center_loss)
                
                cls_loss = sigmoid_focal_loss(cls_pred , cls , alpha=0.25, gamma=2.0, reduction='mean') 
                loss_class += cls_loss.item()
                run['cls_loss'].log(cls_loss)
                
                batch_loss = cls_loss*wclass + center_loss*wcenter + regression_loss*wregress
                run['batch_loss'].log(batch_loss)
                scaler.scale(batch_loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                epoch_loss += batch_loss
                print(f"Batch Loss ---> {batch_loss}")
                print(f"Batch cls Loss ---> {cls_loss}")
                print(f"Batch center Loss ---> {center_loss}")
                print(f"Batch regression Loss ---> {regression_loss}")
                run['lr'].log(lr_schedule.get_last_lr()[0])
                # if (idx + 1) % accum_steps== 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
           
            if lr_schedule == 'onecyclelr':
                lr_schedule.step()
        if lr_schedule != 'onecyclelr':
            lr_schedule.step()
        print(f"Epoch Loss ---> {epoch_loss/len(train_loader)}")
        run['Epoch Loss'].log(epoch_loss/len(train_loader))
        total_cls_loss = loss_class/len(train_loader)
        total_loss_center = loss_center/len(train_loader)
        total_reg_loss = loss_reg/len(train_loader)
        print(f"Epoch cls Loss ---> {total_cls_loss}")
        print(f"Epoch center Loss ---> {total_loss_center}")
        print(f"Epoch regression Loss ---> {total_reg_loss}")
        # checkpoint_path = os.path.join(out_dir, f'train_{epoch}_checkpoint.pth')
        # torch.save({
        #         'epoch': epoch + 1,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'scheduler_state_dict': lr_schedule.state_dict(),
        #         'train_loss': epoch_loss,
        #     }, checkpoint_path)
        
            
        print("-----------------------")
        print(f"Validation for epoch {epoch}")
        model.eval()
        total_accuracy , total_iou = 0 , 0
        for idx , batch in enumerate(val_loader):
            print(f"Current batch : {idx}")
            img , regression , cls , centerness = batch
            img = img.to(device)
            regression = regression.to(device)
            cls = cls.to(device)
            centerness = centerness.to(device)
            
            with torch.no_grad() , autocast():
                
                cls_pred , reg_pred , center_pred = model(img)
            # Calculate metrics
                # classification post process 
                cls_pred = torch.sigmoid(cls_pred)
                accuracy = calculate_accuracy(cls_pred, cls)
                total_accuracy+=accuracy
                run['accuracy'].log(accuracy)
                
                pred_boxes , gt_boxes = extract_bounding_boxes(pred = reg_pred , ground_truth = regression)
                iou = calculate_iou(pred_boxes , gt_boxes)
                ious = iou.mean()
                total_iou+=ious
                # ious.append(avg_batch_iou)
                precision_05, recall_05 = calculate_precision_recall(pred_boxes, gt_boxes, 0.5)
                precision_07, recall_07 = calculate_precision_recall(pred_boxes, gt_boxes, 0.7)
                precision_09, recall_09 = calculate_precision_recall(pred_boxes, gt_boxes, 0.9)
                
                # Log metrics
                run['IoU'].log(ious)
                run['precision@0.5'].log(precision_05)
                run['recall@0.5'].log(recall_05)
                run['precision@0.7'].log(precision_07)
                run['recall@0.7'].log(recall_07)
                run['precision@0.9'].log(precision_09)
                run['recall@0.9'].log(recall_09)
        avg_acc = total_accuracy / len(val_loader)  
        avg_iou = total_iou / len(val_loader)   
        if avg_iou > saver or avg_acc > saver_acc :
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
            saver = avg_iou
            saver_acc = avg_acc
        log_json = {
            'epoch': epoch,
            'lr': lr_schedule.get_last_lr()[0],
            'train cls loss': float(total_cls_loss),
            'train center loss': float(total_loss_center),
            'train regression loss': float(total_reg_loss),
            'Train total loss': float(epoch_loss),
            'Val Accuracy': float(avg_acc),
            'Val IoU': float(avg_iou)
        }
        log_file_path = os.path.join(out_dir, f'log.json')
        with open(log_file_path, 'a') as f:
            json.dump(log_json , f)
            f.write('\n')
        print(f"Validation for epoch {epoch} completed.")
    run.stop()
 
 
 
# def calculate_accuracy(pred_cls, true_cls):
#     pred_cls = torch.argmax(pred_cls, dim=1)
#     correct = (pred_cls == true_cls).float()
#     accuracy = correct.sum() / correct.numel()
#     return accuracy

def calculate_centerness_mse(pred_centerness, true_centerness):
    mse = F.mse_loss(pred_centerness, true_centerness)
    return mse        
def test(model , test_loader , checkpoint , device): 
        
    assert checkpoint != None
    weight = torch.load(checkpoint)
    model.load_state_dict(weight['model_state_dict'])

    model.eval()
    
    for idx , batch in enumerate(test_loader):
        print(f"Current batch : {idx}")
        img , regression , cls , centerness = batch
        img = img.to(device)
        regression = regression.to(device)
        cls = cls.to(device)
        centerness = centerness.to(device)
        
        with torch.no_grad():
            
            cls_pred , reg_pred , center_pred = model(img)
            print(f"cls_pred {cls_pred.shape} , gt_cls {cls.shape}")
            print(f"regression_pred {reg_pred.shape} , gt_regression {regression.shape}")
            print(f"centerness_pred {center_pred.shape} , gt_center {centerness.shape}")
            accuracy = calculate_accuracy(cls_pred , cls)
            print(f"accuracy : {accuracy}")   
                