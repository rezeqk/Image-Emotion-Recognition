import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
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
    
    if model_variant == 'Variant1CNN':
        model = Variant1CNN()
    elif model_variant == 'Variant2CNN':
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
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    return all_labels, all_predictions

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

def get_data_loaders_by_group(dataset_path, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    
    data_loaders = {}
    classes = ['neutral', 'angry', 'happy', 'focused']
    demographics = ['female', 'male', 'middle_aged', 'senior', 'young']
    
    for class_name in classes:
        for demo_name in demographics:
            indices = [i for i, (path, _) in enumerate(dataset.samples) if class_name in path and demo_name in path]
            if indices:
                subset = Subset(dataset, indices)
                data_loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
                data_loaders[(class_name, demo_name)] = data_loader
    
    return data_loaders

def calculate_metrics(labels, predictions):
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    return accuracy, precision, recall, f1

def bias_analysis(model, data_loaders, device):
    results = []
    
    for (class_name, demo_name), data_loader in data_loaders.items():
        labels, predictions = predict_on_dataset(model, data_loader, device)
        accuracy, precision, recall, f1 = calculate_metrics(labels, predictions)
        results.append((class_name, demo_name, accuracy, precision, recall, f1))
    
    return results

if __name__ == "__main__":
    model_path = 'best_model.pth'

    if len(sys.argv) > 1 and sys.argv[1] == "image":
        image_path = sys.argv[2]
        model, device = load_model(model_path)
        prediction = predict_on_image(model, image_path, device)
        print(f"Predicted class for the image: {prediction}")
    else:
        dataset_path = 'dataset'
        model, device = load_model(model_path)
        data_loaders = get_data_loaders_by_group(dataset_path)

        results = bias_analysis(model, data_loaders, device)
        
        headers = ["Class", "Demographic", "Accuracy", "Precision", "Recall", "F1-Score"]
        print(tabulate(results, headers=headers, tablefmt="grid"))

        # Optionally, save results to a CSV file
        import csv
        with open('bias_analysis_results.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(results)
