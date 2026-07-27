import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
#project root folder
def get_dataset_path():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    dataset_path = os.path.join(BASE_DIR, "dataset_v2")
    print("Dataset path:", dataset_path)
    return dataset_path
def load_images_and_labels(dataset_path):
    # Store images and labels
    images = []
    labels = []
    #label mapping
    label_map = {}
    folders = sorted(os.listdir(dataset_path))

    for index, folder in enumerate(folders):
        label_map[folder] = index

    for label in label_map:
        folder_path = os.path.join(dataset_path, label)
   
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
        
            image = cv2.imread(image_path)
            if image is None:
                continue
            image = cv2.resize(image, (64, 64))
            images.append(image)
            labels.append(label_map[label])
    return images, labels
def preprocess_data(images, labels):
    #convert to numpy arrays
    x = np.array(images)
    y = np.array(labels)
    print("Images shape:", x.shape)
    print("Labels shape:", y.shape)
    print("unique labels:", np.unique(y))
    #normalize images
    x = x.astype("float32") / 255.0
    print("Pixel Range:")
    print("Minimum:", x.min())
    print("Maximum:", x.max())
    return x, y
def split_data(x, y):
    #split into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y,shuffle=True)
    #verify split
    print("Training images:", x_train.shape)
    print("Testing images:", x_test.shape)
    print("Training labels:", y_train.shape)
    print("Testing labels:" , y_test.shape)
    return x_train, x_test, y_train, y_test

def load_data():

    dataset_path = get_dataset_path()

    images, labels = load_images_and_labels(dataset_path)

    x, y = preprocess_data(images, labels)

    x_train, x_test, y_train, y_test = split_data(x, y)

    return x_train, x_test, y_train, y_test
