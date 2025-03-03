import torch
import numpy as np
import onnxruntime as ort
import cv2
import os
from PIL import Image
from model import Segmentor
from utils import PostProcessing
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def load_model():
#     model = Segmentor()
#     checkpoint = torch.load(
#         "/home/hasanmog/CNN-VS-ViT/Semantic-Segmentation/CNN-BASED/weights/BEST/best_checkpoint.pth",
#         map_location=device
#     )
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.to(device)
#     model.eval()
#     return model
import onnxruntime as ort

def load_model():
    onnx_model_path = os.path.join(os.path.dirname(__file__), "weights", "best_checkpoint.onnx")
    session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    return session

session = load_model()  

def predict(img, postprocessing=None):
    """Performs inference on an input image and returns the segmentation mask and overlay."""
    
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8))
    elif not isinstance(img, Image.Image):
        raise ValueError("Unsupported input type. Expected an image or NumPy array.")

    
    transform = transforms.Compose([
        transforms.Resize((512, 512)), 
        transforms.ToTensor()
    ])
    
    image = transform(img).unsqueeze(0).numpy().astype(np.float32)
 
    
    print([inp.name for inp in session.get_inputs()])
    onnx_inputs = {"image_input": image}
    onnx_outputs = session.run(None, onnx_inputs)

    outputs = onnx_outputs[0].squeeze(0)
   
    if postprocessing:
        postprocess = PostProcessing()
        outputs = postprocess.post_process_batch(outputs)
        outputs = postprocess.noise_filter(outputs, mina=10).squeeze(0).squeeze(0)
    else:
        outputs = np.squeeze(outputs)

   
    mask_normalized = (outputs * 255).astype(np.uint8) 
    colored_mask = cv2.applyColorMap(mask_normalized, cv2.COLORMAP_VIRIDIS)
    
  
    mask_image = Image.fromarray(cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB))

    
    img_resized = img.resize(mask_image.size)  
    overlay = Image.blend(img_resized, mask_image, alpha=0.5)

    return np.array(mask_image), np.array(overlay)




