import os
import cv2
import numpy as np
#project root folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
dataset_path = os.path.join(BASE_DIR, "dataset")
print("Dataset path:", dataset_path)
# Store images and labels
images = []
labels = []
#label mapping
label_map = {
    "A": 0,
    "B": 1,
    "C": 2
}
for label in label_map:
    folder_path = os.path.join(dataset_path, label)
    print("Processing folder:", folder_path)
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        print("Processing image:", image_path)
        image = cv2.imread(image_path)
        if image is None:
            continue
        image = cv2.resize(image, (64, 64))
        images.append(image)
        labels.append(label_map[label])
#convert to numpy arrays
x = np.array(images)
y = np.array(labels)
print("Images shape:", x.shape)
print("Labels shape:", y.shape)
print("Number of classes:", len(label_map))
print("Number of samples:", len(images))
print("unique labels:", np.unique(y))

