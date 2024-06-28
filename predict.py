import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from tabulate import tabulate
from PIL import Image
from collections import Counter
from cnn_models import Variant1CNN, Variant2CNN, FacialStateCNN

class_names = ["happy", "neutral", "angry", "focus"]

def load_model(model_path):
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
    
    return class_names[prediction]

def get_data_loaders_by_group(dataset, batch_size=64):
    data_loaders = {}
    classes = ['happy', 'neutral', 'angry', 'focus']
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
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    return accuracy, precision, recall, f1, support

def bias_analysis(model, data_loaders, device):
    results = []
    
    for (class_name, demo_name), data_loader in data_loaders.items():
        labels, predictions = predict_on_dataset(model, data_loader, device)
        accuracy, precision, recall, f1, support = calculate_metrics(labels, predictions)
        results.append((class_name, demo_name, accuracy, precision, recall, f1, support))
    
    return results

def print_confusion_matrix(labels, predictions, class_name, demo_name):
    cm = confusion_matrix(labels, predictions, labels=np.arange(len(class_names)))
    print(f"\nConfusion Matrix for {class_name} - {demo_name}:")
    print(tabulate(cm, headers=class_names, showindex=class_names, tablefmt="grid"))

def average_metrics_by_gender(results):
    female_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'support': []}
    male_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'support': []}
    
    for result in results:
        class_name, demo_name, accuracy, precision, recall, f1, support = result
        if 'female' in demo_name:
            if accuracy is not None: female_metrics['accuracy'].append(accuracy)
            if precision is not None: female_metrics['precision'].append(precision)
            if recall is not None: female_metrics['recall'].append(recall)
            if f1 is not None: female_metrics['f1'].append(f1)
            if support is not None: female_metrics['support'].append(support)
        elif 'male' in demo_name:
            if accuracy is not None: male_metrics['accuracy'].append(accuracy)
            if precision is not None: male_metrics['precision'].append(precision)
            if recall is not None: male_metrics['recall'].append(recall)
            if f1 is not None: male_metrics['f1'].append(f1)
            if support is not None: male_metrics['support'].append(support)
    
    avg_female_metrics = {k: np.mean([x for x in v if x is not None]) for k, v in female_metrics.items()}
    avg_male_metrics = {k: np.mean([x for x in v if x is not None]) for k, v in male_metrics.items()}
    
    return avg_female_metrics, avg_male_metrics

def analyze_class_distribution(dataset):
    class_counts = Counter([label for _, label in dataset])
    print("\nClass distribution in the dataset:")
    for class_idx, count in class_counts.items():
        print(f"{class_names[class_idx]}: {count}")

if __name__ == "__main__":
    model_path = 'best_model.pth'
    dataset_path = 'newdataset'

    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = datasets.ImageFolder(dataset_path, transform=transform)

    if len(sys.argv) > 1 and sys.argv[1] == "image":
        image_path = sys.argv[2]
        model, device = load_model(model_path)
        prediction = predict_on_image(model, image_path, device)
        print(f"Predicted class for the image: {prediction}")
    else:
        model, device = load_model(model_path)
        data_loaders = get_data_loaders_by_group(dataset)

        results = []
        
        for (class_name, demo_name), data_loader in data_loaders.items():
            labels, predictions = predict_on_dataset(model, data_loader, device)
            accuracy, precision, recall, f1, support = calculate_metrics(labels, predictions)
            results.append((class_name, demo_name, accuracy, precision, recall, f1, support))
            print_confusion_matrix(labels, predictions, class_name, demo_name)
        
        headers = ["Class", "Demographic", "Accuracy", "Precision", "Recall", "F1-Score", "Support"]
        print(tabulate(results, headers=headers, tablefmt="grid"))

        avg_female_metrics, avg_male_metrics = average_metrics_by_gender(results)
        
        print("\nAverage Metrics for Females:")
        print(tabulate([[avg_female_metrics['accuracy'], avg_female_metrics['precision'], avg_female_metrics['recall'], avg_female_metrics['f1'], avg_female_metrics['support']]], 
                       headers=["Accuracy", "Precision", "Recall", "F1-Score", "Support"], tablefmt="grid"))
        
        print("\nAverage Metrics for Males:")
        print(tabulate([[avg_male_metrics['accuracy'], avg_male_metrics['precision'], avg_male_metrics['recall'], avg_male_metrics['f1'], avg_male_metrics['support']]], 
                       headers=["Accuracy", "Precision", "Recall", "F1-Score", "Support"], tablefmt="grid"))
      
        analyze_class_distribution(dataset)
