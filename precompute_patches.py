import os
import json
import cv2

from utils.candidate_detection import detect_candidate_regions, extract_candidate_patches


dataset_dir = "/content/drive/MyDrive/GCP_Assignment_Datasets/train_dataset"
annotation_file = dataset_dir + "/gcp_marks.json"

output_dir = "/content/patch_dataset"

os.makedirs(output_dir, exist_ok=True)


with open(annotation_file) as f:
    labels = json.load(f)


for img_path, label in labels.items():

    full_path = os.path.join(dataset_dir, img_path)

    img = cv2.imread(full_path)

    if img is None:
        continue

    x_gt = int(label["mark"]["x"])
    y_gt = int(label["mark"]["y"])

    candidates = detect_candidate_regions(img)

    patches, boxes = extract_candidate_patches(img, candidates)

    for i,(x1,y1,x2,y2) in enumerate(boxes):

        if x1 <= x_gt <= x2 and y1 <= y_gt <= y2:

            patch = patches[i]

            patch = cv2.resize(patch,(256,256))

            filename = img_path.replace("/","_") + ".jpg"

            cv2.imwrite(os.path.join(output_dir,filename),patch)

            break