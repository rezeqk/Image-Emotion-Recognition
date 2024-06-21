import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from tabulate import tabulate
import sys
from PIL import Image
from cnn_models import Variant1CNN, Variant2CNN, FacialStateCNN

def load_model(model_path):
    # Check if CUDA is available and set the device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    model_variant = checkpoint['model_variant']
    
    if model_variant == 'variant1':
        model = Variant1CNN()
    elif model_variant == 'variant2':
        model = Variant2CNN()
    elif model_variant == 'FacialStateCNN':
        model = FacialStateCNN()
    else:
        raise ValueError(f"Unknown model variant: {model_variant}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, device

def predict_on_dataset(model, data_loader, device):
    model.to(device)
    predictions = []
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
    return predictions

def predict_on_image(model, image_path, device):
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    print(f"Opening image at path: {image_path}")
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(dim=1).item()
    
    return prediction

if __name__ == "__main__":
    model_path = 'best_model.pth'

    if len(sys.argv) > 1 and sys.argv[1] == "image":
        image_path = sys.argv[2]
        model, device = load_model(model_path)
        prediction = predict_on_image(model, image_path, device)
        print(f"Predicted class for the image: {prediction}")
    else:
        dataset_path = 'dataset'
        transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.ImageFolder(dataset_path, transform=transform)
        data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        model, device = load_model(model_path)
        predictions = predict_on_dataset(model, data_loader, device)
        print("Predictions on dataset:", predictions)
