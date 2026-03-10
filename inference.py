import os
import cv2
import torch

from models.gcp_model import GCPModel
from utils.candidate_detection import detect_candidate_regions, extract_candidate_patches


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GCPModel()

model.load_state_dict(torch.load("/content/gcp_pose_estimation/models/gcp_model.pth", map_location=device))
model.to(device)

model.eval()


def predict_image(image):

    candidates = detect_candidate_regions(image)

    patches, boxes = extract_candidate_patches(image, candidates)

    best_score = -1
    best_result = None

    for i, patch in enumerate(patches):

        patch = cv2.resize(patch,(256,256))

        patch = patch / 255.0

        patch = torch.tensor(patch,dtype=torch.float32).permute(2,0,1)

        patch = patch.unsqueeze(0).to(device)

        coords, shape_logits = model(patch)

        score = torch.max(shape_logits).item()

        if score > best_score:

            best_score = score

            x1,y1,x2,y2 = boxes[i]

            x_local = coords[0][0].item()

            y_local = coords[0][1].item()

            x_global = x1 + x_local
            y_global = y1 + y_local

            shape_class = torch.argmax(shape_logits).item()

            best_result = (x_global, y_global, shape_class)

    return best_result