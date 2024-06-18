import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import sys
from cnn_variants import FacialStateCNN, Variant1CNN, Variant2CNN

def load_model(model_class, model_path):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model_class().to(device)
    model=model_class()
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
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(dim=1).item()
    
    return prediction

if __name__ == "__main__":
    model_path = 'best_model.pth'
    model_class = FacialStateCNN  # Default model

    if len(sys.argv) > 1 and sys.argv[1] == "image":
        image_path = sys.argv[2]
        model = load_model(model_class, model_path)
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
        model = load_model(model_class, model_path)
        predictions = predict_on_dataset(model, data_loader)
        print("Predictions on dataset:", predictions)