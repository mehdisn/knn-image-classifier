import os
import cv2
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder):
    """Load images from a folder and return data and labels."""
    image_paths = list(paths.list_images(folder))
    data = []
    labels = []

    for image_path in image_paths:
        label = os.path.basename(os.path.dirname(image_path))
        labels.append(label)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        data.append(img)

    data = np.array(data)
    labels = np.array(labels)

    le = LabelEncoder()
    labels = le.fit_transform(labels)

    return data, labels

def preprocess_data(data):
    """Flatten the image data."""
    dataset_size = data.shape[0]
    data = data.reshape(dataset_size, -1)
    return data

def split_data(data, labels, test_size=0.25, random_state=42):
    """Split data into train and test sets."""
    return train_test_split(data, labels, test_size=test_size, random_state=random_state)
