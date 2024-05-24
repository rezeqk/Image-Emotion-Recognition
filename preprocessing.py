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

#task 1 
def plot_class_distribution(dataset_path, title):
    # Load the dataset
    dataset = datasets.ImageFolder(dataset_path)
    
    # Calculate class distribution
    class_counts = {class_name: 0 for class_name in dataset.classes}
    for _, label in dataset.imgs:
        class_name = dataset.classes[label]
        class_counts[class_name] += 1
    
    # Data for plotting
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    # Create a bar chart
    plt.figure(figsize=(10, 8))
    plt.bar(classes, counts, color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

directories = [r'train', r'test']
for directory in directories:
    plot_class_distribution(directory, f'Class Distribution in {directory.capitalize()} Dataset')

#task 2
def plot_pixel_intensity_distribution(loader, title='Pixel Intensity Distribution'):
    """
    Plots the pixel intensity distributions for the images in the provided DataLoader.

    Parameters:
        loader (DataLoader): DataLoader containing the images whose pixel intensities are to be plotted.
        title (str): The title of the plot.
    """
    # Initialize a dictionary to store pixel data for each class
    pixel_data = {class_name: [] for class_name in loader.dataset.classes}
    
    # Iterate over batches of images
    for images, labels in loader:
        for image, label in zip(images, labels):
            # Unnormalize the images for analysis
            image = image * 0.5 + 0.5  # Adjust if different normalization was used
            # Convert image tensor to numpy array and flatten it
            pixels = image.numpy().flatten()
            # Append the pixel data to the corresponding class
            class_name = loader.dataset.classes[label]
            pixel_data[class_name].extend(pixels)

    # Plotting the histograms
    fig, axs = plt.subplots(nrows=len(pixel_data), figsize=(10, 8))
    fig.suptitle(title)

    for ax, (class_name, pixels) in zip(axs, pixel_data.items()):
        ax.hist(pixels, bins=30, color='gray', range=(0, 1))
        ax.set_title(class_name)
        ax.set_xlim([0, 1])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

directories = [train_loader, test_loader]
for directory in directories:
    plot_pixel_intensity_distribution(directory, title='Pixel Intensity Distribution')


#task 3
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

for directory in directories:
    plot_sample_pixel_intensity_histograms(directory)