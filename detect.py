import torch
from PIL import Image
import argparse
from pathlib import Path

import torch.nn as nn
import torchvision.transforms as transforms

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformation
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# Define the model architecture
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


def predict_image(image_path, model_path="ather_binary_classifier.pth"):
    """Predict if an image is ather or other using the trained model"""

    # Check if model exists
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}. Please train the model first.")
        return None, 0

    # Load and prepare image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, 0

    # Transform and process image
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Load model
    model = BinaryClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        probability = output.item()

    return probability


def main():
    parser = argparse.ArgumentParser(description="Detect if an image is Ather or other")
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the image to classify"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ather_binary_classifier.pth",
        help="Path to the trained model (default: ather_binary_classifier.pth)",
    )
    args = parser.parse_args()

    probability = predict_image(args.image, args.model)

    if probability > 0.9:
        result = "other"
    else:
        result = "ather"

    print(f"Result: {result}")


if __name__ == "__main__":
    main()
