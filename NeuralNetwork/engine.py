import torch
import neptune
from tqdm import tqdm
from torch.optim import optimizer
from torch.nn.functional import cross_entropy


def train_one_epoch(model , training_loader , validation_loader, optimizer ,loss_func, device ):
    
    batch_loss = 0.0
    train_loss = 0.0
    losses = 0
    model.train(True)
    model.to(device)
    for i , batch in enumerate(training_loader):
        
        images , labels = batch
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = loss_func(outputs , labels) # CE Loss expects the outputs of the model to be the logits and the targets to be the class indices.
        loss.backward()
        
        optimizer.step()
        
        batch_loss+=loss.item()
        
    train_loss = batch_loss / len(training_loader)
    print("Final Training loss:" , train_loss)
    
    batch_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i , batch in enumerate(validation_loader):
            
            images , labels = batch 
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            loss = loss_func(outputs , labels)
            batch_loss += loss.item()
        
        val_loss = batch_loss / len(validation_loader)
        print("Validation Loss:" , val_loss)
            
    return train_loss , val_loss

def test_one_epoch(model , test_loader , loss_func , device):
    
    model.eval()
    test_loss = 0.0
    batch_loss = 0.0
      
    with torch.no_grad:
        for i , batch in enumerate(test_loader):
            
            images , labels = batch 
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            loss = loss_func(outputs , labels)
            batch_loss += loss.item()
        
        test_loss = batch_loss / len(test_loader)
        print("Test Loss:" , test_loss)
        
    return test_loss
            
        
        
        