import numpy as np
from sklearn.utils import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from tabulate import tabulate
import sys
from PIL import Image

class Variant2CNN(nn.Module):
    def __init__(self):
        super(Variant2CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(12 * 12 * 128, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 4)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x



def load_model(model_path):
    model = Variant2CNN()
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()
    return model

def predict_on_dataset(model, data_loader):
    device = next(model.parameters()).device
    predictions = []
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
    return predictions

def predict_on_image(model, image_path):
    device = next(model.parameters()).device
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
        model = load_model(model_path)
        prediction = predict_on_image(model, image_path)
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
        model = load_model(model_path)
        predictions = predict_on_dataset(model, data_loader)
        print("Predictions on dataset:", predictions)
