import torch
import neptune
import numpy as np
from tqdm import tqdm
from torch.optim import optimizer
from torch.nn.functional import cross_entropy

run = neptune.init_run(
    project='Solo/Solo',  # specify your project name here
    api_token= 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzZjFkNjIxNy1jOWZlLTQzZTUtYTA3OS0zMjhlM2UwNTMzODYifQ==',
    #with_id = 'VLMEO-1048'
    )   
def train_one_epoch(model , training_loader , validation_loader, optimizer ,lr_scheduler ,epochs , loss_func, device , out_dir ):
    

    batch_loss = 0.0
    train_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    losses = 0
    model.train(True)
    model.to(device)
    for epoch in tqdm(range(epochs)):
    
        print(f"Epoch number : {epoch + 1}")
    
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
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']  
        run['Learning Rate'].log(current_lr)
        print("current_lr" , current_lr) 
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
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)
                
            accuracy = correct_predictions / total_predictions
            val_loss = batch_loss / len(validation_loader)
            print("Validation Loss:" , val_loss)
            print("Validation Accuracy" , accuracy*100)
            
            if val_loss > losses:
                torch.save({
                    'epoch' : epoch+1 ,
                    'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict' : optimizer.state_dict() , 
                    'train_loss' : train_loss , 
                    'val_loss' : val_loss
                } , out_dir)
            
                losses = val_loss
        run['training loss per Epoch'].log(train_loss)
        run['validation loss per Epoch'].log(val_loss)
        run['validation Accuracy per Epoch'].log(accuracy*100)
        
    run.stop()  
    return train_loss , val_loss , current_lr
# run.stop()
def test_one_epoch(model, test_loader, loss_func, device):
    model.eval()  # Ensure the model is in evaluation mode
    test_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_func(outputs, labels)
            test_loss += loss.item()

            # Convert outputs to probabilities and find the predicted class
            _, predicted_classes = torch.max(outputs, 1)
            correct_predictions += (predicted_classes == labels).sum().item()
            total_samples += labels.size(0)

    average_test_loss = test_loss / len(test_loader)
    accuracy = correct_predictions / total_samples

    print("Test Loss:", average_test_loss)
    print("Accuracy:", accuracy * 100)

    return average_test_loss, accuracy


            
