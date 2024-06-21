import numpy as np
from sklearn.utils.class_weight import compute_class_weight
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
from cnn_models import FacialStateCNN, Variant1CNN, Variant2CNN

# Preprocess the data
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset from directories
dataset = datasets.ImageFolder('dataset', transform=transform)

# Random seed for reproducibility of data
random_seed = 42
torch.manual_seed(random_seed)

# Splitting weights for 70% train, 15% validate, and 15% test
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Split function to split data
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(random_seed))

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Calculate class weights for training
train_labels = [label for _, label in train_dataset]
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# Training function
def train_model(model, train_loader, val_loader, model_variant, num_epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)
    best_val_loss = float('inf')
    patience = 5  # Wait 5 epochs if there's no improvement in loss function early stopping
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
            torch.save({
                'model_variant': model_variant,
                'model_state_dict': model.state_dict()
            }, 'best_model.pth')
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping!')
                break

    return val_losses

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
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())  # If using GPU change back to CPU before converting to numpy

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

results = []
# Loop to train and test the 3 variants
for variant_name, model_class in [("FacialStateCNN", FacialStateCNN), 
                                  ("Variant1CNN", Variant1CNN), 
                                  ("Variant2CNN", Variant2CNN)]:
    print(f"Training {variant_name}...")
    model = model_class()
    val_losses = train_model(model, train_loader, val_loader, variant_name)
    model.load_state_dict(torch.load('best_model.pth')['model_state_dict'])
    conf_matrix, metrics = evaluate_model(model, test_loader)
    results.append((variant_name, metrics))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(f'Confusion Matrix for {variant_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    print(f'{variant_name} Classification Report:')
    print(metrics)

# Print results
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
