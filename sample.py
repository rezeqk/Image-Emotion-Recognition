import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Define the path to the dataset
dataset_path = r'train'

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

# Function to plot images and their histograms
def plot_images_and_histograms(images, class_name):
    fig, axes = plt.subplots(5, 6, figsize=(20, 15))
    axes = axes.ravel()
    
    # Randomly select 15 images
    selected_images = random.sample(images, 15)
    
    for i in range(15):
        # Plot image
        axes[2 * i].imshow(selected_images[i], cmap='gray')
        axes[2 * i].set_title(f'{class_name} Image {i+1}')
        axes[2 * i].axis('off')
        
        # Plot histogram
        axes[2 * i + 1].hist(selected_images[i].ravel(), bins=256, color='gray', alpha=0.7)
        axes[2 * i + 1].set_title(f'{class_name} Histogram {i+1}')
        axes[2 * i + 1].set_xlim([0, 256])
        
    plt.tight_layout()
    plt.show()

# Plot images and histograms for each class
for cls in classes:
    plot_images_and_histograms(images[cls], cls)