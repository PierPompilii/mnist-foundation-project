import os
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model import DigitRecognizerCNN  # Import your CNN model

# ---- Settings ---- #
config = {
    "batch_size": 128,
    "learning_rate": 3e-4,
    "epochs": 10,
    "validation_split": 0.1,
    "checkpoint_dir": Path("checkpoints"),
}

# ---- Make sure folders exist ---- #
config["checkpoint_dir"].mkdir(exist_ok=True)

# ---- Set device (GPU if available) ---- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---- Transform images ---- #
transform = transforms.Compose([
    transforms.RandomRotation(5),                 # Slightly rotate digits
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))    # MNIST mean and std
])

# ---- Load and split MNIST dataset ---- #
dataset = datasets.MNIST(root="data", train=True, transform=transform, download=True)
val_size = int(len(dataset) * config["validation_split"])
train_size = len(dataset) - val_size
train_data, val_data = random_split(dataset, [train_size, val_size])

test_data = datasets.MNIST(root="data", train=False, transform=transform, download=True)

# ---- DataLoaders ---- #
train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_data, batch_size=config["batch_size"])
test_loader = DataLoader(test_data, batch_size=config["batch_size"])

# ---- Initialize model, loss, optimizer ---- #
model = DigitRecognizerCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# ---- Training function ---- #
def train_one_epoch():
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss

# ---- Validation function ---- #
def validate(loader):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            correct += preds.eq(labels).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = 100.0 * correct / len(loader.dataset)
    return avg_loss, accuracy

# ---- Main Training Loop ---- #
best_val_acc = 0
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for epoch in range(1, config["epochs"] + 1):
    print(f"\nEpoch {epoch}/{config['epochs']}")

    train_loss = train_one_epoch()
    val_loss, val_acc = validate(val_loader)

    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_path = config["checkpoint_dir"] / f"best_model_{timestamp}.pth"
        torch.save(model.state_dict(), best_path)
        print(f" New best model saved at: {best_path}")

# ---- Final Test Evaluation ---- #
test_loss, test_acc = validate(test_loader)
print(f"\n Final Test Accuracy: {test_acc:.2f}%")

# ---- Save final model ---- #
final_path = config["checkpoint_dir"] / f"final_model_{timestamp}.pth"
torch.save(model.state_dict(), final_path)
print(f" Final model saved at: {final_path}")
