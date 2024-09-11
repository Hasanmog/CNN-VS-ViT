import torch.optim
from Vit import VisionTransformer
from DataLoader import EuroSAT , UC_MERCED , custom_collate_fn
import argparse
import torch
import os
from torch.utils.data import DataLoader
import pandas as pd
import torch.nn as torch
from torchvision import  transforms


parser = argparse.ArgumentParser(
                    description='Transformer-Based Classifier',)

# Model Args
parser.add_argument('--patch_size' , type = int , default = 16 , help = "size of each patch - default 16")
parser.add_argument('--embed_dim' , type = int , default = 512 ,  help = "embedding dimension - default is 1024")
parser.add_argument('--max_patches' , type = int , default = 10000 , help = "maximum number of patches - default 10,000")
parser.add_argument('--num_heads' , type = int , default = 4 , help = " number of heads of the attention block - default is 4" )
parser.add_argument('--num_layers' , type = int , default = 2 , help = "number of transformer layers - default is 2")

#Dataset Args
parser.add_argument('--dataset' , type = str , required = True , help = " UC_MERCED OR EuroSAT")
parser.add_argument('--parent_dir' , type = str , required=True , help = "path to the folder containing the dataset")
parser.add_argument('--batch_size' , type = int , default = 4 , help = "Batch Size - default is 8")

# hyperparameters
parser.add_argument('--lr' , type = float , default = 1e-3 , help = "learning rate - default is 1e-3")
parser.add_argument('--weight_decay' , type = float , default = 1e-3 , help = "Adam optimizer weight decay - default is 0" )
parser.add_argument('--epochs' , type = int , required=True , help = "Number of training epochs")
parser.add_argument('--use_scheduler' , type = bool , default = True , help = "use exponential decay scheduler - default True")
parser.add_argument('--resume'  , type = str , default = "/home/hasanmog/CNN-VS-ViT/Classification/Transformer-Based/weights/model_epoch_150_val_acc_0.7776.pth" )
parser.add_argument('--save_dir' , type = str , required = True , help = 'directory to save the training checkpoint')
 
args = parser.parse_args()

def build_model(patch_size , 
                            embed_dim ,
                            max_patches , 
                            num_heads , 
                            num_layers  , 
                            num_classes , 
                            device  , 
                          ):
    """
    Prepares and builds the Vision Transformer model for training.

    Args:
        embed_dim (int): The embedding dimension for the transformer layers.
        num_heads (int): The number of attention heads in the multi-head attention mechanism.
        num_layers (int): The number of transformer layers to be stacked.
        mlp_dim (int): The dimension of the MLP (feed-forward) network in each transformer layer.
        device (torch.device): The device to which the model will be moved (e.g., 'cuda' or 'cpu').

    Returns:
        torch.nn.Module: The Vision Transformer model moved to the specified device.
    """
    model = VisionTransformer(patch_size , 
                                                embed_dim ,
                                                max_patches, 
                                                num_heads , 
                                                num_layers ,  
                                                num_classes)
    
    return model.to(device)

def dataset(dataset_name, 
            parent_dir, 
            batch_size):
    """
    Load the dataset (UC_MERCED or EuroSAT) and return the DataLoaders for training, validation, and testing.

    Args:
        dataset_name (str): The name of the dataset ('UC_MERCED' or 'EuroSAT').
        parent_dir (str): The root directory where the datasets are located.
        batch_size (int): The batch size to use for the DataLoader.
    
    Returns:
        tuple: train_loader, val_loader, test_loader, len(class_names)
    """
    
    # Supported datasets
    supported_datasets = ['UC_MERCED', 'EuroSAT']
    assert dataset_name in supported_datasets, f"Unknown Dataset: {dataset_name}. Choose from {supported_datasets}"

    # Define common transforms
    common_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    if dataset_name == 'UC_MERCED':
        class_names = ['Agricultural', 'Airplane', 'Baseball diamond', 'Beach', 'Buildings', 
                       'Chaparral', 'Dense residential', 'Forest', 'Freeway', 'Golf course', 
                       'Harbor', 'Intersection', 'Medium residential', 'Mobile home park', 
                       'Overpass', 'Parking lot', 'River', 'Runway', 'Sparse residential', 
                       'Storage tanks', 'Tennis court']
        
        dataset_paths = {
            'train': os.path.join(parent_dir, "Images/train"),
            'val': os.path.join(parent_dir, "Images/val"),
            'test': os.path.join(parent_dir, "Images/test")
        }
        
        train_set = UC_MERCED(parent_dir=dataset_paths['train'], transform=common_transforms['train'])
        val_set = UC_MERCED(parent_dir=dataset_paths['val'], transform=common_transforms['train'])
        test_set = UC_MERCED(parent_dir=dataset_paths['test'], transform=common_transforms['test'])

    elif dataset_name == 'EuroSAT':
        class_names = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
                       "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        def load_csv_data(file_path):
            return pd.read_csv(file_path).sort_values(by=['ClassName']).to_dict('records')

        dataset_paths = {
            'train': os.path.join(parent_dir, "EuroSAT/train.csv"),
            'val': os.path.join(parent_dir, "EuroSAT/validation.csv"),
            'test': os.path.join(parent_dir, "EuroSAT/test.csv")
        }

        train_set = EuroSAT(parent_dir, data=load_csv_data(dataset_paths['train']), transform=transform)
        val_set = EuroSAT(parent_dir, data=load_csv_data(dataset_paths['val']), transform=transform)
        test_set = EuroSAT(parent_dir, data=load_csv_data(dataset_paths['test']), transform=transform)

    # Helper function to create DataLoader
    def create_loader(dataset, shuffle, batch_size, collate_fn=None):
        return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size , collate_fn= custom_collate_fn)

    # Creating loaders
    train_loader = create_loader(train_set, shuffle=True, batch_size=batch_size)
    val_loader = create_loader(val_set, shuffle=True, batch_size=batch_size)
    test_loader = create_loader(test_set, shuffle=False, batch_size=batch_size // 2)

    print(f"Number of training samples: {len(train_set)}, validation samples: {len(val_set)}, testing samples: {len(test_set)}")
    
    return train_loader, val_loader, test_loader, len(class_names)



import os
import torch

def train(model, dataset, device, lr, weight_decay, epochs, save_dir, use_sched, resume_checkpoint=None):
    """
    Trains the model on the specified dataset, with the option to resume from a checkpoint.

    Args:
        resume_checkpoint (str, optional): Path to the checkpoint to resume from.
    """
    # Load dataset
    train_loader, val_loader, _, num_classes = dataset(
        dataset_name=args.dataset,
        parent_dir=args.parent_dir,
        batch_size=args.batch_size
    )
    
    # Build model
    model = build_model(
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        max_patches=args.max_patches,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_classes=num_classes,
        device=device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    criterion = torch.nn.CrossEntropyLoss()

    # Resume training if a checkpoint is provided
    start_epoch = 0
    best_val_accuracy = 0.0

    if resume_checkpoint:
        print(f"Loading checkpoint from {resume_checkpoint}")
        start_epoch, best_val_accuracy = load_checkpoint(resume_checkpoint, model, optimizer, lr_scheduler)
        print(f"Resumed from epoch {start_epoch+1}, best validation accuracy: {best_val_accuracy:.4f}")

    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (img, labels) in enumerate(train_loader):
            # Move data to device
            img, labels = img.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(img)
        
            # Compute loss
            loss = criterion(logits, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Print progress every 10 batches
            if batch_idx % 10 == 9:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        if use_sched and epoch > 30:
            lr_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{epochs}] - Learning Rate: {current_lr:.6f}")

        # Validation step after each epoch
        val_loss, val_accuracy = validate(model, val_loader, device, criterion)
        print(f"Epoch [{epoch+1}/{epochs}] - Training Loss: {running_loss/len(train_loader):.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        
        # Check if validation accuracy is better than the best seen so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # Save the model checkpoint
            save_checkpoint(model, optimizer, epoch, best_val_accuracy, loss, save_dir)
            print(f"Checkpoint saved at epoch {epoch+1} with validation accuracy: {val_accuracy:.4f}")


def validate(model, val_loader, device, criterion):
    """
    Validates the model on the validation set.

    Args:
        model (torch.nn.Module): The model to be validated.
        val_loader (DataLoader): The validation data loader.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') to validate the model on.
        criterion (torch.nn.Module): The loss function.

    Returns:
        float: Validation loss.
        float: Validation accuracy.
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for img, labels in val_loader:
            img, labels = img.to(device), labels.to(device)
            
            img = img.to(device)
            labels = labels.to(device)
            # Forward pass
            logits = model(img)
            # Compute loss
            loss = criterion(logits, labels)
            val_loss += loss.item()
            
            # Compute accuracy
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = val_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def save_checkpoint(model, optimizer, epoch, val_accuracy,loss, save_dir):
    """
    Saves the model checkpoint to the specified directory, including the learning rate.

    Args:
        model (torch.nn.Module): The model to be saved.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): The epoch number at which the model is saved.
        val_accuracy (float): The validation accuracy at the time of saving.
        save_dir (str): Directory where the checkpoint will be saved.

    Returns:
        None
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get the current learning rate (assuming there's only one param group)
    current_lr = optimizer.param_groups[0]['lr']

    checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch+1}_val_acc_{val_accuracy:.4f}.pth")

    # Save the model, optimizer state, and learning rate
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss' : loss , 
        'val_accuracy': val_accuracy,
        'learning_rate': current_lr  # Include learning rate in the checkpoint
    }, checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer, lr_scheduler=None):
    """
    Loads the model and optimizer state from a checkpoint, and resumes the training process.

    Args:
        checkpoint_path (str): Path to the saved checkpoint file.
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        lr_scheduler (torch.optim.lr_scheduler, optional): The learning rate scheduler to restore.

    Returns:
        int: The last epoch saved in the checkpoint (to resume training from).
        float: The best validation accuracy saved in the checkpoint.
    """
    checkpoint = torch.load(checkpoint_path)
    
    # Load the model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load the optimizer state dict
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Optionally restore the learning rate scheduler
    if lr_scheduler:
        for param_group in optimizer.param_groups:
            param_group['lr'] = checkpoint.get('learning_rate', param_group['lr'])
    
    # Return the last epoch and best validation accuracy
    return checkpoint['epoch'], checkpoint['val_accuracy']



def main():
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    
    # Load the dataset
    print("Loading dataset...")
    train_loader, val_loader, test_loader, num_classes = dataset(
        dataset_name=args.dataset,
        parent_dir=args.parent_dir,
        batch_size=args.batch_size
    )
    
    # Build the model
    print("Building model...")
    model = build_model(
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        max_patches=args.max_patches,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_classes=num_classes,
        device=device
    )
    
    # Train the model
    print("Starting training...")
    train(
        model=model,
        dataset=lambda dataset_name, parent_dir, batch_size: (train_loader, val_loader, test_loader, num_classes),
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        save_dir=args.save_dir , 
        use_sched=args.use_scheduler,
        resume_checkpoint=args.resume

    )


if __name__ == '__main__':
    main()

            
            
    
    

