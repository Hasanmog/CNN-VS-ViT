import sys
import os
import gradio as gr
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(parent_dir)
import torch
import numpy as np
from Classifiers.Model import NN_4
from torchvision import  transforms
from torchvision.transforms import ToTensor
from PIL import Image

# Define the transformation
# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts image to tensor and scales to [0, 1]
    transforms.Resize((64,64)) , 
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalizes the tensor
])

def preprocess(image_input):
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Check if the input is a file path or a numpy array
        if isinstance(image_input, str):
            # It's a file path
            img = Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            # It's a numpy array, convert to PIL Image
            img = Image.fromarray(image_input.astype('uint8'), 'RGB')
        else:
            raise ValueError("Unsupported input type")

        img = transform(img)  # Apply the transformations
        img = img.to(device)  # Move the tensor to the GPU if available
        return img
    except Exception as e:
        print(f"An error occurred while preprocessing the image: {e}")
        return None

def load_model():
    model = NN_4()  # Create an instance of the model
    weights_path = 'Classifiers/weights/NN_attempt_4/best_checkpoint.pth'  # Path to the model weights
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("/home/hasanmog/paper2code/Classifiers/weights/NN_attempt_4/best_checkpoint.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Set the model to inference mode
    return model


def run(input):
    try:
        classes = ['Annual Crop', 'Forest', 'Herbaceous Vegetation', 'Highway', 'Industrial',
                   'Pasture', 'Permanent Crop', 'Residential', 'River', 'SeaLake']
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        processed_input = preprocess(input)
        if processed_input is None:
            return "Error processing image"

        processed_input = processed_input.unsqueeze(0)
        model = load_model()
        model.eval()
        with torch.no_grad():
            logits = model(processed_input).to('cpu')
        logits = logits.numpy()
        pred_classes = np.argmax(logits, axis=1)
        pred_class = classes[pred_classes[0]]
        return pred_class
    except Exception as e:
        return f"An error occurred during classification: {e}"

    
iface = gr.Interface(fn=run, 
                     inputs="image", 
                     outputs="label" , 
                     title = "Classification Model trained on EuroSAT dataset (64 x 64)" ,
                     description="Please upload an image of size 64x64 pixels. This model is specifically trained on images of this size for optimal performance")

# Run the interface
iface.launch()