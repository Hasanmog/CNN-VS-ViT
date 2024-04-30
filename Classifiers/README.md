# EuroSAT Image Classification with Deep Learning

## Project Overview

This project develops a deep learning model to classify satellite images from the EuroSAT dataset into 10 different land use and land cover classes. By leveraging convolutional neural networks (CNNs), the model achieves high accuracy, demonstrating the potential of CNNs in remote sensing applications for geographical image processing.

## Model Architecture

The model architecture is designed to efficiently process 2D satellite images and capture spatial hierarchies for accurate classification. Below is a detailed overview of the model's layers and parameters:

### Architecture Summary

```plaintext
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 62, 62]             448
         MaxPool2d-2           [-1, 16, 31, 31]               0
       BatchNorm2d-3           [-1, 16, 31, 31]              32
              ReLU-4           [-1, 16, 31, 31]               0
            Conv2d-5           [-1, 32, 29, 29]           4,640
       BatchNorm2d-6           [-1, 32, 29, 29]              64
              ReLU-7           [-1, 32, 29, 29]               0
            Conv2d-8           [-1, 64, 27, 27]          18,496
       BatchNorm2d-9           [-1, 64, 27, 27]             128
       LogSoftmax-10           [-1, 64, 27, 27]               0
           Conv2d-11          [-1, 128, 26, 26]          32,896
      BatchNorm2d-12          [-1, 128, 26, 26]             256
             ReLU-13          [-1, 128, 26, 26]               0
           Conv2d-14          [-1, 256, 24, 24]         295,168
        MaxPool2d-15            [-1, 256, 8, 8]               0
      BatchNorm2d-16            [-1, 256, 8, 8]             512
             ReLU-17            [-1, 256, 8, 8]               0
          Flatten-18                [-1, 16384]               0
           Linear-19                   [-1, 64]       1,048,640
           Linear-20                   [-1, 32]           2,080
           Linear-21                   [-1, 16]             528
           Linear-22                   [-1, 10]             170
       LogSoftmax-23                   [-1, 10]               0
================================================================
Total params: 1,404,058
Trainable params: 1,404,058
Non-trainable params: 0
----------------------------------------------------------------

## Results Summary

The deep learning model trained on the EuroSAT dataset demonstrated excellent performance across both validation and test sets. Here's a summary of the key performance metrics:

### Performance Metrics

- **Training Loss:** `0.03`
- **Validation Loss:** `0.56`
- **Validation Accuracy:** `88.81%`

- **Test Loss:** `0.557`
- **Test Accuracy:** `89.33%`

```

### Analysis

The results indicate that the model is capable of accurately classifying different land use and land cover types from satellite images with high accuracy. The relatively low training and validation losses suggest that the model has learned the underlying patterns in the data effectively, without overfitting.

## Experiment Logs

For detailed logs of the classification attempts, including hyperparameter variations and outcomes, see the [View Attempts Log](Attempts.txt). This log provides insights into the iterative process of model tuning and performance evaluation.
