import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from tabulate import tabulate
from cnn_models import FacialStateCNN, Variant1CNN, Variant2CNN

#preprocess the data
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#load dataset from directories
dataset = datasets.ImageFolder('dataset', transform=transform)

#random seed for reproducibility of data
random_seed = 42
torch.manual_seed(random_seed)

#class names
class_names = ["happy", "neutral", "angry", "focus"]

#training function
def train_model(model, train_loader, val_loader, model_variant, class_weights_tensor, num_epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)
    best_val_loss = float('inf')
    patience = 5  # Wait 5 epochs if there's no improvement in loss function early stopping
    trigger_times = 0

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
        scheduler.step(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping!')
                break

    model.load_state_dict(best_model_state)
    return model

#evaluation function
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
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    class_metrics = precision_recall_fscore_support(all_labels, all_preds, zero_division=0)
    macro_metrics = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    micro_metrics = precision_recall_fscore_support(all_labels, all_preds, average='micro', zero_division=0)

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

#kfold function
def k_fold_cross_validation(model_class, dataset, k=10, num_epochs=50):
    kf = KFold(n_splits=k, shuffle=True, random_state=random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_fold_metrics = []
    all_class_reports = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        print(f"Starting fold {fold + 1}/{k}")

        train_val_idx = train_idx[:int(0.85 * len(train_idx))]
        val_idx = train_idx[int(0.85 * len(train_idx)):]

        train_subset = Subset(dataset, train_val_idx)
        val_subset = Subset(dataset, val_idx)
        test_subset = Subset(dataset, test_idx)

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

        #calculate class weights for training
        train_labels = [label for _, label in train_subset]
        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

        model = model_class().to(device)
        model = train_model(model, train_loader, val_loader, model_class.__name__, class_weights_tensor, num_epochs)
        conf_matrix, metrics = evaluate_model(model, test_loader)
        all_fold_metrics.append(metrics)
        all_class_reports.append(metrics['class_report'])

        #plot confusion matrix
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix for {model_class.__name__} - Fold {fold+1}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

        print(f"Fold {fold + 1}/{k} metrics: {metrics}")

    #compute average metrics across all folds
    avg_metrics = {
        "accuracy": np.mean([metrics["accuracy"] for metrics in all_fold_metrics]),
        "macro_precision": np.mean([metrics["macro_precision"] for metrics in all_fold_metrics]),
        "macro_recall": np.mean([metrics["macro_recall"] for metrics in all_fold_metrics]),
        "macro_f1": np.mean([metrics["macro_f1"] for metrics in all_fold_metrics]),
        "micro_precision": np.mean([metrics["micro_precision"] for metrics in all_fold_metrics]),
        "micro_recall": np.mean([metrics["micro_recall"] for metrics in all_fold_metrics]),
        "micro_f1": np.mean([metrics["micro_f1"] for metrics in all_fold_metrics]),
    }

    avg_class_report = {
        "precision": np.mean([cr["precision"] for cr in all_class_reports], axis=0),
        "recall": np.mean([cr["recall"] for cr in all_class_reports], axis=0),
        "f1": np.mean([cr["f1"] for cr in all_class_reports], axis=0),
        "support": np.sum([cr["support"] for cr in all_class_reports], axis=0)
    }

    avg_metrics["class_report"] = avg_class_report

    return avg_metrics, all_fold_metrics

if __name__ == "__main__":
    results = []
    # Loop to train and test the 3 variants with k-fold cross-validation
    for variant_name, model_class in [("FacialStateCNN", FacialStateCNN),
                                      ("Variant1CNN", Variant1CNN),
                                      ("Variant2CNN", Variant2CNN)]:
        print(f"Performing k-fold cross-validation for {variant_name}...")
        avg_metrics, all_fold_metrics = k_fold_cross_validation(model_class, dataset, k=10, num_epochs=50)
        results.append((variant_name, avg_metrics, all_fold_metrics))
        print(f"Avg metrics for {variant_name}: {avg_metrics}")

    # Print and compare results
    print("Model\tMacro Precision\tMacro Recall\tMacro F1\tMicro Precision\tMicro Recall\tMicro F1\tAccuracy")
    for result in results:
        name, avg_metrics, _ = result
        print(f"{name}\t{avg_metrics['macro_precision']:.4f}\t{avg_metrics['macro_recall']:.4f}\t{avg_metrics['macro_f1']:.4f}\t"
              f"{avg_metrics['micro_precision']:.4f}\t{avg_metrics['micro_recall']:.4f}\t{avg_metrics['micro_f1']:.4f}\t{avg_metrics['accuracy']:.4f}")
        print(f"{name} per class:")
        class_report = avg_metrics["class_report"]
        table = []
        for i, (prec, rec, f1, support) in enumerate(zip(class_report["precision"], class_report["recall"], class_report["f1"], class_report["support"])):
            table.append([class_names[i], prec, rec, f1, support])
        headers = ["Class", "Precision", "Recall", "F1", "Support"]
        print(tabulate(table, headers=headers, tablefmt="grid"))
