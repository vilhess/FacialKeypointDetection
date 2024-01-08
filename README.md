# Nose Detection using Fine-tuned ResNet

## Overview

This project focuses on nose detection in images using deep learning techniques. The model is based on a pre-trained ResNet architecture, which has been fine-tuned for the specific task of nose detection.

## Prerequisites

Before running the code, ensure you have the correct dependencies installed using:

`pip install -r requirements.txt`

## Dataset

The dataset used for training and testing the model is the [Facial Key Point Detection Dataset](https://www.kaggle.com/datasets/prashantarorat/facial-key-point-data?rvi=1)

## Training and Testing

To train the model, run the following command:

`python training.py`

don't forget to specify the DEVICE at the top of the file and set TRAINING to True. Else, the model will be loaded from the specified path and will be tested on the test set.

## Results

In the folder saved_images, you can find the results of the model on the test set for a particular image during the training.