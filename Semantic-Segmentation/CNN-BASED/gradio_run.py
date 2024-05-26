import gradio as gr
import torch
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import Segmentor
from utils import PostProcessing
from torchvision import transforms
from matplotlib.colors import Normalize




def load_model():
    model = Segmentor()  # Create an instance of the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("/home/hasanmog/CNN-VS-ViT/Semantic-Segmentation/CNN-BASED/weights/BEST/best_checkpoint.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Set the model to inference mode
    return model

def predict(img ,postprocessing , device = 'cuda'):
    if isinstance(img, np.ndarray):
        # Convert numpy array (H, W, C) in the range [0, 255] to a PIL Image
        img = Image.fromarray(img.astype(np.uint8))
    elif not isinstance(img, Image.Image):
        raise ValueError("Unsupported input type. Expected an image.")
    # img = Image.open(image_path)
    model = load_model()
    transform = transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor()])
    image = transform(img)
    image = image.to(device)
    with torch.no_grad():
        logits = model(image.unsqueeze(0)) #add batch dim
        outputs = torch.sigmoid(logits).cpu().numpy()
        if postprocessing:
            postprocess = PostProcessing()
            outputs = postprocess.post_process_batch(outputs)
            outputs = postprocess.noise_filter(outputs , mina = 10).squeeze(0).squeeze(0)
        else: 
            outputs = outputs.squeeze(0).squeeze(0)
        pred = outputs
    norm = Normalize(vmin=pred.min(), vmax=pred.max())
    colormap = plt.get_cmap('viridis')
    colored_mask = (colormap(norm(pred))[:, :, :3] * 255).astype(np.uint8)  # Ignore alpha
    mask_image = Image.fromarray(colored_mask)

    # Create an overlay image
    img = img.resize((512 , 512))
    mask_image = mask_image.resize(size=img.size)
    overlay = Image.blend(img, mask_image, alpha=0.5)

    return  np.array(mask_image), np.array(overlay)  
iface = gr.Interface(fn=predict,
                     inputs=["image",
                              gr.Checkbox(label="Apply Postprocessing")],
                     outputs=["image", "image"] , 
                     title="Building Segmentation Model",
    description=""" This a CNN-based model trained on WHU Building Dataset (512 x 512) , where it acheived 72% IoU and 83% F1 score on the test set.
                                        
                                        please check the PostProcessing box to apply postprocessing on the output.
                                        
                                        Note that Postprocessing sets a threshold of 0.5 on the output masks , with noise filtering.
                                        """
)
iface.launch()

