import torch
import torch.nn.functional as F


def compute_loss(pred_boxes , gt_boxes , pred_class , gt_class , obj_score ):
    
    w1 = 1 # class weight
    w2 = 1 # object_Score weight
    w3 = 1 # box weight
    class_loss = F.cross_entropy(pred_class , gt_class , reduction = 'none')
    
    


def train(model , train_loader , val_loader , epochs , out_dir , device):
    
    model.to(device)
    
    