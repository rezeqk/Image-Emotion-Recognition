from torchvision import datasets
import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(dataset_path, title):
    # Load the dataset
    dataset = datasets.ImageFolder(dataset_path)
    
    class_counts = {class_name: 0 for class_name in dataset.classes}
    for _, label in dataset.imgs:
        class_name = dataset.classes[label]
        class_counts[class_name] += 1
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=classes, y=counts, palette='viridis')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

directories = [r'train', r'test']
for directory in directories:
    plot_class_distribution(directory, f'Class Distribution in {directory.capitalize()} Dataset')
