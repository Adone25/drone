import os
import json
import cv2

from inference import predict_image


test_dir = "/content/drive/MyDrive/GCP_Assignment_Datasets/test_dataset"

shape_map = {
    0:"Cross",
    1:"Square",
    2:"L-Shaped"
}

results = {}

for root, dirs, files in os.walk(test_dir):

    for file in files:

        if file.lower().endswith(".jpg"):

            img_path = os.path.join(root,file)

            image = cv2.imread(img_path)

            pred = predict_image(image)

            if pred is None:
                continue

            x,y,shape_id = pred

            rel_path = os.path.relpath(img_path, test_dir)

            results[rel_path] = {
                "mark":{
                    "x": float(x),
                    "y": float(y)
                },
                "verified_shape": shape_map[shape_id]
            }


with open("predictions.json","w") as f:

    json.dump(results,f,indent=4)


print("predictions.json generated")