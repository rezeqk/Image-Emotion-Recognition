import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Define the path to the dataset
dataset_path = r'train'  # Use raw string to avoid escape sequence issues

# Define the classes
classes = ['Happy', 'Neutral', 'Angry', 'Focused']

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

# Calculate frequency of each pixel intensity
intensity_counts = {cls: Counter(intensities) for cls, intensities in pixel_intensities.items()}

# Plot the pixel intensity distributions using bar charts
plt.figure(figsize=(14, 8))
for cls, counts in intensity_counts.items():
    keys = list(counts.keys())
    values = list(counts.values())
    plt.bar(keys, values, label=cls, alpha=0.7, width=1.0)  # Adjust alpha for better visualization if overlapping

plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Aggregated Pixel Intensity Distribution by Class')
plt.legend()
plt.show()
