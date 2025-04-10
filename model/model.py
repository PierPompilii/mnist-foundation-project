import torch
import torch.nn as nn
import torch.nn.functional as F

# A simple CNN for recognizing handwritten digits (like from MNIST)
class DigitRecognizerCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # First convolution layer:
        # - 1 input channel (grayscale image)
        # - 10 output channels
        # - 5x5 kernel size
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

        # Second convolution layer:
        # - Takes the 10 outputs from the previous layer
        # - Outputs 20 feature maps
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        # Fully connected layer after flattening
        self.fc1 = nn.Linear(320, 50)  # 20 channels * 4 * 4 = 320 features
        self.fc2 = nn.Linear(50, 10)   # 10 output classes (digits 0–9)

        # Dropout layer to help prevent overfitting
        self.dropout = nn.Dropout(p=0.3)
        self.dropout2d = nn.Dropout2d(p=0.5)

    def forward(self, x):
        # Apply first conv layer, then ReLU, then max pooling
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        # Apply second conv layer, ReLU, max pooling, and dropout
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.dropout2d(x)

        # Flatten the output to feed into fully connected layers
        x = x.view(-1, 320)

        # First fully connected layer + ReLU
        x = F.relu(self.fc1(x))

        # Dropout only if we're training
        if self.training:
            x = self.dropout(x)

        # Final layer gives us the scores for each digit class
        x = self.fc2(x)

        return x