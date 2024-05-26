import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to the dataset
dataset_path = r'train'  # Use raw string to avoid escape sequence issues

# Define the classes
classes = ['Happy', 'Neutral', 'Angry']

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

# Initialize a dictionary to hold pixel intensities for each class
pixel_intensities = {cls: [] for cls in classes}

# Aggregate pixel intensities for each class
for cls, imgs in images.items():
    for img in imgs:
        pixel_intensities[cls].extend(img.flatten())

# Plot the pixel intensity distributions
plt.figure(figsize=(14, 8))
for cls, intensities in pixel_intensities.items():
    sns.histplot(intensities, bins=256, kde=True, label=cls, stat='density')

plt.xlabel('Pixel Intensity')
plt.ylabel('Density')
plt.title('Aggregated Pixel Intensity Distribution by Class')
plt.legend()
plt.show()
