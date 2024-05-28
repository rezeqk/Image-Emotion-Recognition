import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

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

def plot_sample_pixel_intensity_histograms(loader, num_samples_per_class=15):
    """
    Plots histograms of pixel intensities for sample images from each class.

    Parameters:
        loader (DataLoader): DataLoader containing the images whose pixel intensities are to be plotted.
        num_samples_per_class (int): Number of sample images to plot per class.
    """
    class_samples = {class_name: [] for class_name in loader.dataset.classes}
    
    # Collect samples for each class
    for images, labels in loader:
        for image, label in zip(images, labels):
            class_name = loader.dataset.classes[label]
            if len(class_samples[class_name]) < num_samples_per_class:
                class_samples[class_name].append(image)
            if all(len(samples) >= num_samples_per_class for samples in class_samples.values()):
                break
        if all(len(samples) >= num_samples_per_class for samples in class_samples.values()):
            break
    
    # Plotting the images and histograms for each class
    for class_name, samples in class_samples.items():
        fig, axs = plt.subplots(2, num_samples_per_class, figsize=(15, 5))
        fig.suptitle(f'Images and Pixel Intensity Distributions for {class_name} Samples')
        
        for i, image in enumerate(samples):
            # Unnormalize the images for display
            img = image * 0.5 + 0.5
            # Convert image tensor to numpy array
            img_np = img.numpy().squeeze()
            
            # Plot image
            axs[0, i].imshow(img_np, cmap='gray')
            axs[0, i].axis('off')
            axs[0, i].set_title(f'Sample {i+1}')
            
            # Convert image tensor to numpy array and flatten it for histogram
            pixels = img.numpy().flatten()
            # Plot histogram
            axs[1, i].hist(pixels, bins=30, color='gray', range=(0, 1))
            axs[1, i].set_xlim([0, 1])
            axs[1, i].set_title(f'Sample {i+1}')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

directories = [train_loader, test_loader]

for directory in directories:
    plot_sample_pixel_intensity_histograms(directory)