#  MNIST Digit Recognizer - Model & Training Description

This project contains a beginner-friendly implementation of a CNN trained on the MNIST dataset. The goal is to learn the core concepts of deep learning, practice model training, and use the model inside a simple web app.

---

##  Model Architecture

A simple convolutional neural network (CNN) using PyTorch. It includes two convolutional layers, dropout for regularization, and two fully connected layers for classification.

```python
class DigitRecognizerCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.dropout = nn.Dropout(p=0.3)
        self.dropout2d = nn.Dropout2d(p=0.5)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.dropout2d(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        if self.training:
            x = self.dropout(x)
        x = self.fc2(x)
        return x
```

###  What Each Layer Does:
- **Conv Layers**: Detect patterns in the image like edges and corners.
- **Max Pooling**: Reduces the image size and keeps the most important features.
- **Dropout**: Helps avoid overfitting by randomly ignoring neurons during training.
- **Fully Connected Layers**: Take extracted features and classify them into one of the 10 digits.

---

##  Training Process

###  Settings:
- **Dataset**: MNIST (28x28 grayscale images of digits 0–9)
- **Batch Size**: 128
- **Learning Rate**: 0.0003
- **Epochs**: 10
- **Loss Function**: `CrossEntropyLoss` (ideal for multi-class classification)
- **Optimizer**: `Adam` (good default choice for most tasks)
- **Device**: Automatically uses GPU if available

###  Data Augmentation Used:
```python
transform = transforms.Compose([
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```
- **RandomRotation(5)**: Makes the model more robust to slight drawing angle differences.
- **Normalization**: Matches the mean and std of the MNIST dataset.

---

##  Training Loop Summary
1. Load MNIST and split into training/validation
2. Train the model for a few epochs
3. Evaluate accuracy on the validation set after each epoch
4. Save the best model checkpoint if validation accuracy improves
5. After training ends, test the model on the test set and save final weights

---

##  Output

- **Best Model**: Saved during training to `/checkpoints/best_model_YYYYMMDD_HHMMSS.pth`
- **Final Model**: Always saved at the end as `final_model_YYYYMMDD_HHMMSS.pth`

These files can later be loaded into the Streamlit app to make predictions.

---

##  Why It Works (Even If It's Simple)

- The architecture is small but powerful enough for MNIST
- Dropout helps generalize to real inputs
- Rotation augmentation helps match what users draw by hand
- Adam optimizer and CrossEntropyLoss work well together for this task

This project is a great starting point before moving on to more advanced architectures and datasets.

