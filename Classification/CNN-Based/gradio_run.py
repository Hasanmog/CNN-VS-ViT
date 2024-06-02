import sys
import os
import gradio as gr
# current_dir = os.path.dirname(__file__)
# parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
# sys.path.append(parent_dir)
import torch
import numpy as np
from Model import NN_4
from torchvision import  transforms
from torchvision.transforms import ToTensor
from PIL import Image

# Define the transformation
# Define the transformation
transform = transforms.Compose([
    transforms.Resize((256,256)) , 
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("/home/hasanmog/CNN-VS-ViT/Classifiers/CNN-Based/weights/UC-Merced/UC-Merced.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Set the model to inference mode
    return model


def run(input):
    try:
        classes = ['Agricultural', 'Airplane', 'Baseball diamond', 'Beach', 'Buildings', 'Chaparral', 'Dense residential', 'Forest', 'Freeway',
               'Golf course', 'Harbor', 'Intersection', 'Medium residential', 'Mobile home park', 'Overpass', 'Parking lot', 'River',
               'Runway', 'Sparse residential', 'Storage tanks', 'Tennis court']
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

    
iface = gr.Interface(
    fn=run,
    inputs="image",
    outputs="label",
    title="Land Cover Classification Model",
    description="""
    Welcome to the Land Cover Classification Model interface! This tool uses a deep learning model trained on the UC-Merced dataset, 
    specifically designed to recognize 21 different types of land cover from satellite images. Each image should be 256x256 pixels in size 
    for optimal classification accuracy. The model achieved an accuracy of 84% on the test set, showcasing its effectiveness in diverse environments.

    Please upload a satellite image of size 256x256 pixels to classify it into one of the following categories:
    - Agricultural
    - Airplane
    - Baseball Diamond
    - Beach
    - Buildings
    - Chaparral
    - Dense Residential
    - Forest
    - Freeway
    - Golf Course
    - Harbor
    - Intersection
    - Medium Residential
    - Mobile Home Park
    - Overpass
    - Parking Lot
    - River
    - Runway
    - Sparse Residential
    - Storage Tanks
    - Tennis Court

    After uploading an image, the model will predict the category it believes best represents the land cover seen in the image.
    """
)

# Run the interface
iface.launch()