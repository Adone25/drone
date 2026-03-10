# Aerial GCP Pose Estimation

## Project Overview
This project builds a computer vision pipeline to detect Ground Control Points (GCPs) in aerial drone images.

The model performs two tasks:
1. Predict marker center coordinates (x, y)
2. Classify marker shape (Cross, Square, L-Shaped)

## Pipeline
Image
↓
Candidate Detection (OpenCV)
↓
Patch Extraction
↓
CNN Model (ResNet18)
↓
Coordinate + Shape Prediction

## Training

Run training using:

python train.py

## Inference

Generate predictions using:

python generate_predictions.py

Output file:

predictions.json

## Technologies Used
- Python


- PyTorch
- OpenCV

## Model Weights

The trained model weights can be downloaded here:

https://drive.google.com/file/d/1Tm8NBwh-oSyfpI3cAYosRGV6bzthNuif/view?usp=sharing

Place the file in:

models/gcp_model.pth
- Google Colab

## Author
Adone Jose
