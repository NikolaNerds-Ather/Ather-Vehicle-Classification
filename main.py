import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform_train = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


data_dir = Path("dataset")


full_dataset = torchvision.datasets.ImageFolder(
    root=str(data_dir), transform=transform_train
)


dataset_size = len(full_dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size


train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42),
)


test_dataset.dataset.transform = transform_test
val_dataset.dataset.transform = transform_test


batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# print(
#     f"Dataset split complete | Train: {train_size} | Validation: {val_size} | Test: {test_size}"
# )


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


model = BinaryClassifier().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            f"Epoch [{epoch + 1}/{epochs}] | Loss: {running_loss / len(train_loader):.4f}"
            f"Accuracy: {100 * correct / total:.2f}%"
        )


def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    return 100 * correct / total


def predict_image(image_path, model_path="ather_binary_classifier.pth"):
    image = Image.open(image_path).convert("RGB")

    image_tensor = transform_test(image).unsqueeze(0).to(device)

    model = BinaryClassifier().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        output = model(image_tensor)
        probability = output.item()
        prediction = 1 if probability > 0.5 else 0

    # class_names = list(full_dataset.class_to_idx.keys())
    # class_name = class_names[prediction]

    # print(f"Prediction: {class_name}")
    # print(f"Probability: {probability:.4f}")

    return prediction, probability


def test_with_custom_image(image_path):
    if not Path("ather_binary_classifier.pth").exists():
        print("Model not found. Training and saving model first...")

        num_epochs = 10
        train(num_epochs)

        # accuracy = test()

        # torch.save(model.state_dict(), "ather_binary_classifier.pth")
        # print(f"Model saved. Final accuracy: {accuracy:.2f}%")

    prediction, probability = predict_image(image_path)

    if probability > 0.9:
        return "other"
    else:
        return "ather"


if __name__ == "__main__":
    # num_epochs = 10
    # train(num_epochs)
    # accuracy = test()
    # torch.save(model.state_dict(), "ather_binary_classifier.pth")
    # print(f"Model saved. Final accuracy: {accuracy:.2f}%")

    parser = argparse.ArgumentParser(description="Classify images as Ather or other")
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the image to classify"
    )
    args = parser.parse_args()

    res = test_with_custom_image(args.image)
    print(f"Result: {res}")
