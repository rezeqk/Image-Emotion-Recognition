import matplotlib.pyplot as plt
from cnn_variants import FacialStateCNN, Variant1CNN, Variant2CNN 
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

# Define transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))
])

# Load the dataset from the directory
dataset = datasets.ImageFolder('dataset', transform=transform)

# Set random seed for reproducibility
random_seed = 42
torch.manual_seed(random_seed)

#splitting weights for 70% train, 15% validate, and 15% test 
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

#split
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(random_seed))

#data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

train_classes = [label for _, label in train_dataset]
class_weights = compute_class_weight('balanced', classes=np.unique(train_classes), y=train_classes)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)



def find_learning_rate(model_class, train_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_class()
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-7)  # start with a very small learning rate
    lr_find_epochs = 2
    lr_increase_factor = 1.05  # increase LR by 5% each batch

    lrs = []
    losses = []
    best_loss = float('inf')

    model.train()
    for epoch in range(lr_find_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if loss < best_loss:
                best_loss = loss
            if loss > 4 * best_loss:
                break
            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])
            loss.backward()
            optimizer.step()
            optimizer.param_groups[0]['lr'] *= lr_increase_factor

    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.show()

    find_learning_rate(FacialStateCNN, train_loader)