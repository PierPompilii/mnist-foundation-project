import torch
import torch.nn as nn
import torch.optim as optim 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 
from model import CNN  # model is in model.py

#set device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#prepare the dataset
transform = transforms.ToTensor()
train_data = datasets.MNIST(root="data", train=True, transform=transform, download=True)
test_data = datasets.MNIST(root="data", train=False, transform=transform, download=True)

loaders = {
    "train": DataLoader(train_data, batch_size=100, shuffle=True, num_workers=0),
    "test": DataLoader(test_data, batch_size=100, shuffle=False, num_workers=0)
}

#instantiate the model, the optimize and the loss
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

#training function
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loaders["train"]):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loaders['train'].dataset)} "
                  f"({100. * batch_idx / len(loaders['train']):.0f}%)]\tLoss: {loss.item():.6f}")
            

#testing function
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loaders["test"]:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loaders["test"].dataset)
    accuracy = 100. * correct / len(loaders["test"].dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loaders['test'].dataset)} ({accuracy:.2f}%)\n")


#run training
if __name__ == "__main__":
    for epoch in range(1, 11):
        train(epoch)
        test()

#save the model
torch.save(model.state_dict(), "model/mnist_cnn_model.pth")
print("Model saved to model/mnist_cnn_model.pth")