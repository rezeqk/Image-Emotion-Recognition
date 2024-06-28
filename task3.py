import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.Resize((48, 48)),                 
    transforms.Grayscale(),                       
    transforms.ToTensor(),                         
    transforms.Normalize((0.5,), (0.5,))           
])

train_dataset = datasets.ImageFolder('newdataset', transform=transform)
test_dataset = datasets.ImageFolder('newdataset_3', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

def plot_sample_pixel_intensity_histograms(loader, num_samples_per_class=15):
    class_samples = {class_name: [] for class_name in loader.dataset.classes}
    #collect samples for each class
    for images, labels in loader:
        for image, label in zip(images, labels):
            class_name = loader.dataset.classes[label]
            if len(class_samples[class_name]) < num_samples_per_class:
                class_samples[class_name].append(image)
            if all(len(samples) >= num_samples_per_class for samples in class_samples.values()):
                break
        if all(len(samples) >= num_samples_per_class for samples in class_samples.values()):
            break
    #plot the images and histograms for each class
    for class_name, samples in class_samples.items():
        fig, axs = plt.subplots(2, num_samples_per_class, figsize=(15, 5))
        fig.suptitle(f'Images and Pixel Intensity Distributions for {class_name} Samples')
        
        for i, image in enumerate(samples):
            img = image * 0.5 + 0.5
            img_np = img.numpy().squeeze()
            #plot image
            axs[0, i].imshow(img_np, cmap='gray')
            axs[0, i].axis('off')
            axs[0, i].set_title(f'Sample {i+1}')
        
            pixels = img.numpy().flatten()
            #plot histogram
            axs[1, i].hist(pixels, bins=30, color='gray', range=(0, 1))
            axs[1, i].set_xlim([0, 1])
            axs[1, i].set_title(f'Sample {i+1}')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

directories = [train_loader, test_loader]
for directory in directories:
    plot_sample_pixel_intensity_histograms(directory)