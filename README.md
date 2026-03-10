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
- Google Colab

## Author
Adone Jose
