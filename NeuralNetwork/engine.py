import torch
from torch.optim import optimizer
from torch.nn.functional import cross_entropy

def train_one_epoch(model , training_loader , optimizer ,loss_func, device):
    
    batch_loss = 0.0
    epoch_loss = 0.0
    
    model.to(device)
    for i , batch in enumerate(training_loader):
        
        images , labels = batch
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        pred_class = torch.argmax(outputs , dim = 1)
        print("labels:" , labels)
        print("pred_class : " , pred_class)
        loss = loss_func(pred_class , labels)
        loss.backward()
        
        optimizer.step()
        
        batch_loss+=loss.item()
        
    epoch_loss = batch_loss / len(training_loader)
                 
    return epoch_loss
        
        
        
        
        