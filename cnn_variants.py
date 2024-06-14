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


class FacialStateCNN(nn.Module):
    def __init__(self):
        super(FacialStateCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
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
            nn.Linear(512, 4)  # 4 classes: happy, neutral, focused, angry
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layer(x)
        return x
    

class Variant1CNN(nn.Module):
    def __init__(self):
        super(Variant1CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(12 * 12 * 256, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 4)  # 4 classes
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

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
            nn.Linear(512, 4)  # 4 classes
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

# Training function
def train_model(model, train_loader, val_loader,  num_epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)
    best_val_loss = float('inf')
    patience = 5
    trigger_times = 0
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}')


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping!')
                break

# Evaluation function
def evaluate_model(model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    class_metrics = precision_recall_fscore_support(all_labels, all_preds)
    macro_metrics = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    micro_metrics = precision_recall_fscore_support(all_labels, all_preds, average='micro')

    class_report = {
        "precision": class_metrics[0],
        "recall": class_metrics[1],
        "f1": class_metrics[2],
        "support": class_metrics[3]
    }

    return confusion_matrix(all_labels, all_preds), {
        "accuracy": accuracy,
        "macro_precision": macro_metrics[0],
        "macro_recall": macro_metrics[1],
        "macro_f1": macro_metrics[2],
        "micro_precision": micro_metrics[0],
        "micro_recall": micro_metrics[1],
        "micro_f1": micro_metrics[2],
        "class_report": class_report
    }

# Train and evaluate each variant
results = []

for variant_name, model_class in [("Main Model", FacialStateCNN), 
                                  ("Variant 1", Variant1CNN), 
                                  ("Variant 2", Variant2CNN)]:
    print(f"Training {variant_name}...")
    model = model_class()
    val_losses = train_model(model, train_loader, val_loader)
    model.load_state_dict(torch.load('best_model.pth'))
    conf_matrix, metrics = evaluate_model(model, test_loader)
    results.append((variant_name, metrics))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(f'Confusion Matrix for {variant_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    print(f'{variant_name} Classification Report:')
    print(metrics)
    

# Print the results in the required table format
print("Model\tMacro Precision\tMacro Recall\tMacro F1\tMicro Precision\tMicro Recall\tMicro F1\tAccuracy")
for result in results:
    name, metrics = result
    print(f"{name}\t{metrics['macro_precision']:.4f}\t{metrics['macro_recall']:.4f}\t{metrics['macro_f1']:.4f}\t"
          f"{metrics['micro_precision']:.4f}\t{metrics['micro_recall']:.4f}\t{metrics['micro_f1']:.4f}\t{metrics['accuracy']:.4f}")
    print(f"{name} per class:")
    class_report = metrics["class_report"]
    table = []
    for i, (prec, rec, f1, support) in enumerate(zip(class_report["precision"], class_report["recall"], class_report["f1"], class_report["support"])):
        table.append([i, prec, rec, f1, support])
    headers = ["Class", "Precision", "Recall", "F1", "Support"]
    print(tabulate(table, headers=headers, tablefmt="grid"))