# CNN-Based Semantic Segmentation of Buildings

This repository contains a CNN-based model designed for the semantic segmentation of buildings, utilizing the WHU Building Dataset. The model is capable of identifying building footprints from aerial imagery, which is crucial for urban planning and remote sensing applications.

## Dataset

The model is trained and tested on the [**WHU Building Dataset**](http://gpcv.whu.edu.cn/data/building_dataset.html), which is a well-known dataset in the remote sensing community for building detection. It consists of high-resolution aerial images, making it suitable for training deep learning models that require detailed spatial features.

## Model Architecture

Below is the architecture of the CNN-based model for semantic segmentation:

```plaintext
Layer (type)        Output Shape         Param #
==================================================
Conv2d-1         [-1, 64, 510, 510]       1,792
ReLU-2           [-1, 64, 510, 510]           0
BatchNorm2d-3    [-1, 64, 510, 510]         128
Conv2d-4         [-1, 96, 508, 508]      55,392
ReLU-5           [-1, 96, 508, 508]           0
BatchNorm2d-6    [-1, 96, 508, 508]         192
MaxPool2d-7      [-1, 96, 254, 254]           0
Conv2d-8         [-1, 128, 252, 252]    110,720
ReLU-9           [-1, 128, 252, 252]          0
BatchNorm2d-10   [-1, 128, 252, 252]        256
Conv2d-11        [-1, 256, 250, 250]    295,168
ReLU-12          [-1, 256, 250, 250]          0
BatchNorm2d-13   [-1, 256, 250, 250]        512
MaxPool2d-14     [-1, 256, 125, 125]          0
Conv2d-15        [-1, 256, 62, 62]      590,080
ReLU-16          [-1, 256, 62, 62]            0
ConvTranspose2d-17  [-1, 128, 124, 124] 131,200
Conv2d-18        [-1, 128, 122, 122]    147,584
ReLU-19          [-1, 128, 122, 122]          0
BatchNorm2d-20   [-1, 128, 122, 122]        256
ConvTranspose2d-21  [-1, 64, 244, 244]  32,832
Conv2d-22        [-1, 64, 242, 242]     36,928
ReLU-23          [-1, 64, 242, 242]           0
BatchNorm2d-24   [-1, 64, 242, 242]         128
ConvTranspose2d-25  [-1, 1, 242, 242]        65
Upsample-26      [-1, 1, 512, 512]            0
==================================================
Total params: 1,403,233
Trainable params: 1,403,233
Non-trainable params: 0
--------------------------------------------------
Input size (MB): 3.00
Forward/backward pass size (MB): 1769.01
Params size (MB): 5.35
Estimated Total Size (MB): 1777.36
--------------------------------------------------
```

## Evaluation Results

The model was evaluated using standard metrics such as IoU (Intersection over Union) and F1 Score. 

**with postprocessing**, which consist of thresholding(0.5) and noise filtering:
- **IoU**: 72%
- **F1 Score**: 83%

**Without Postprocessing**:
- **IoU@0.3**: 68%
- **IoU@0.5**: 71.06%
- **IoU@0.75**: 67%
- **Precision** : 71%
- **Recall** : 67.28%
- **F1 Score**: 82%

## Inference Results

Below are some example results from the model, showing the input images, ground truth masks, and the predictions by the model:

![Inference Results](.asset/output.png)

## Weight File :
You can find the training weight file [here](https://drive.google.com/file/d/1i7zgGVAvLdjQe93rtwE9O11Q6wWrWJ3O/view?usp=sharing)

## WorkArea:
The `WorkArea.ipynb` is a notebook where I've done the training and inference.
## Gradio Web Interface

A Gradio web interface has been set up to allow users to easily test the model with their own images or use sample images from the dataset. This interface provides a user-friendly way to interact with the model and visualize the segmentation results in real-time.

### Running the Gradio Interface

To run the Gradio interface, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Hasanmog/CNN-VS-ViT

   cd Semantic-Segmentation/CNN-BASED

   pip install -r requirements.txt

2. **gradio_run.py** :
     
     replace `line 18` with the path of the donwloaded checkpoint.

3. **Run**
