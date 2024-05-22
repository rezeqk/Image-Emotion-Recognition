import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to the dataset
dataset_path = r'train'


# Define the classes
classes = ['Happy', 'Neutral' , 'Angry']

# Function to load images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

# Load images into a dictionary
images = {cls: load_images_from_folder(os.path.join(dataset_path, cls)) for cls in classes}

# Count the number of images per class
class_counts = {cls: len(imgs) for cls, imgs in images.items()}

# Create a bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), palette='viridis')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Class Distribution')
plt.show()