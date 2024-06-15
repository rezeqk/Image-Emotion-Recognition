import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt  #
import numpy as np
from collections import Counter

#brebrocessing the data
transform = transforms.Compose([
    transforms.Resize((48, 48)),                   
    transforms.Grayscale(),                        
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))           
])

train_dataset = datasets.ImageFolder('train', transform=transform)
test_dataset = datasets.ImageFolder('test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#bixel Intensity Distribution
def plot_pixel_intensity_distribution(loader, title='Pixel Intensity Distribution'):
    pixel_data = {class_name: Counter() for class_name in loader.dataset.classes}
    for images, labels in loader:
        for image, label in zip(images, labels):
            # Unnormalize the images for analysis
            image = (image * 0.5 + 0.5).numpy()  # Convert to numpy array
            pixels = image.flatten()
            class_name = loader.dataset.classes[label]
            pixel_data[class_name].update(pixels)
    fig, axs = plt.subplots(len(pixel_data), 1, figsize=(10, len(pixel_data)*4), tight_layout=True)
    fig.suptitle(title)
    for ax, (class_name, counts) in zip(axs.flatten(), pixel_data.items()):
        keys = np.array(list(counts.keys()))
        values = np.array(list(counts.values()))
        ax.bar(keys, values, width=0.005, color='gray') 
        ax.set_title(class_name)
        ax.set_xlim([0, 1])
    plt.show()

#loading the data to access the folders
directories = [train_loader, test_loader]
for directory in directories:
    plot_pixel_intensity_distribution(directory, title='Pixel Intensity Distribution')
