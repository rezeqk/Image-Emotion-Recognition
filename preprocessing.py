import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt  # Corrected import statement
import numpy as np
from collections import Counter

# Define transformations
transform = transforms.Compose([
    transforms.Resize((48, 48)),                   # Resize the image to 48x48
    transforms.Grayscale(),                        # Convert the image to grayscale
    transforms.ToTensor(),                         # Convert the image to a PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))           # Normalize the tensor
])

# Load the dataset from directories
train_dataset = datasets.ImageFolder('train', transform=transform)
test_dataset = datasets.ImageFolder('test', transform=transform)

# Set up the data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Task 2 - Plot Pixel Intensity Distribution Using Bars
def plot_pixel_intensity_distribution(loader, title='Pixel Intensity Distribution'):
    """
    Plots the pixel intensity distributions for the images in the provided DataLoader using individual bars for each pixel intensity.

    Parameters:
        loader (DataLoader): DataLoader containing the images whose pixel intensities are to be plotted.
        title (str): The title of the plot.
    """
    # Initialize a dictionary to store pixel data for each class
    pixel_data = {class_name: Counter() for class_name in loader.dataset.classes}
    
    # Iterate over batches of images
    for images, labels in loader:
        for image, label in zip(images, labels):
            # Unnormalize the images for analysis
            image = (image * 0.5 + 0.5).numpy()  # Convert to numpy array
            pixels = image.flatten()
            class_name = loader.dataset.classes[label]
            pixel_data[class_name].update(pixels)

    # Create a figure with subplots
    fig, axs = plt.subplots(len(pixel_data), 1, figsize=(10, len(pixel_data)*4), tight_layout=True)
    fig.suptitle(title)

    # Plotting individual bars for each class
    for ax, (class_name, counts) in zip(axs.flatten(), pixel_data.items()):
        keys = np.array(list(counts.keys()))
        values = np.array(list(counts.values()))
        ax.bar(keys, values, width=0.005, color='gray')  # Ensure width is small enough to show individual bars
        ax.set_title(class_name)
        ax.set_xlim([0, 1])

    plt.show()

# Example usage
directories = [train_loader, test_loader]
for directory in directories:
    plot_pixel_intensity_distribution(directory, title='Pixel Intensity Distribution')
