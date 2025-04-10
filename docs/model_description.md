#  MNIST CNN Model Description

This section explains the architecture and training process of the Convolutional Neural Network (CNN) used in this project to classify handwritten digits from the MNIST dataset.

---

##  Model Architecture

The model is a simple yet effective CNN implemented using PyTorch. It is designed to identify digits (0 through 9) from 28x28 pixel grayscale images.

### Code Summary (from `model.py`):
```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)       # Output: [batch, 10, 24, 24]
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)      # Output: [batch, 20, 20, 20]
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))         # [batch, 10, 12, 12]
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))  # [batch, 20, 4, 4]
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
```

###  Beginner-Friendly Line-by-Line Explanation

```python
class CNN(nn.Module):
```
➡️ Defines a custom neural network by inheriting from `nn.Module` — the base class for all PyTorch models.

```python
    def __init__(self):
        super(CNN, self).__init__()
```
➡️ Initializes the model. `super()` sets up the necessary internal functions.

```python
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
```
➡️ First convolutional layer:
- Input: 1 channel (grayscale image)
- Output: 10 channels (feature maps)
- Kernel: 5x5 filter to scan for patterns

```python
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
```
➡️ Second convolutional layer:
- Input: 10 feature maps from `conv1`
- Output: 20 feature maps
- Learns more complex features

```python
        self.conv2_drop = nn.Dropout2d()
```
➡️ Dropout layer that randomly disables features during training for regularization.

```python
        self.fc1 = nn.Linear(320, 50)
```
➡️ Fully connected layer:
- Input: flattened output from conv layers (320 units)
- Output: 50 units for abstraction

```python
        self.fc2 = nn.Linear(50, 10)
```
➡️ Final output layer:
- 10 outputs — one for each digit (0–9)

```python
    def forward(self, x):
```
➡️ Defines how data flows through the network.

```python
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
```
➡️ Apply first conv layer → max pooling → ReLU activation.

```python
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
```
➡️ Apply second conv layer → dropout → max pooling → ReLU.

```python
        x = x.view(-1, 320)
```
➡️ Flatten the 2D output into a 1D vector for the fully connected layer.

```python
        x = F.relu(self.fc1(x))
```
➡️ Pass through first dense layer with ReLU activation.

```python
        x = F.dropout(x, training=self.training)
```
➡️ Apply dropout again during training.

```python
        x = self.fc2(x)
```
➡️ Output raw scores (logits) for the 10 digit classes.

```python
        return x
```
➡️ Return the final prediction scores (to be used with `CrossEntropyLoss`).

---

##  Layer-by-Layer Breakdown

| Layer        | Type           | Shape (Approx.)       | Purpose |
|--------------|----------------|------------------------|---------|
| `conv1`      | Conv2D (1→10)  | 28×28 → 24×24          | Extracts local features using 10 filters of size 5×5 |
| `max_pool`   | MaxPool2D      | 24×24 → 12×12          | Reduces spatial dimensions, keeps strong features |
| `conv2`      | Conv2D (10→20) | 12×12 → 8×8            | Learns more abstract features |
| `dropout`    | Dropout2D      |                        | Regularization: randomly turns off neurons |
| `max_pool`   | MaxPool2D      | 8×8 → 4×4              | Further reduces size |
| `flatten`    | Reshape        | 4×4×20 → 320           | Prepares tensor for fully connected layer |
| `fc1`        | Linear         | 320 → 50               | Learns high-level combinations |
| `dropout`    | Dropout        |                        | Prevents overfitting |
| `fc2`        | Linear         | 50 → 10                | Final class scores for 10 digits |

---

##  Training Details

- **Loss Function:** `CrossEntropyLoss()` – combines Softmax and Negative Log Likelihood
- **Optimizer:** `Adam()` with a learning rate of 0.001
- **Epochs:** 10
- **Batch Size:** 100
- **Device:** Trained on CPU or GPU (automatically selected)
- **Dataset:** MNIST – 60,000 training and 10,000 test grayscale images (28x28)

---

##  Model Output

The final layer outputs raw **logits** (unnormalized class scores) for the 10 digit classes. The prediction is made using:
```python
pred = output.argmax(dim=1)
```
This selects the class with the highest score.

---

##  Why This Works

- **Convolutional layers** detect shapes, strokes, and patterns
- **Pooling layers** reduce overfitting and computation
- **Dropout** improves generalization
- **Fully connected layers** map extracted features to class scores

This architecture, though simple, consistently achieves >98% accuracy on MNIST with proper training.

---

##  Output File

- The trained model is saved to:
  ```
  mnist_cnn_model.pth
  ```
  using:
```python
torch.save(model.state_dict(), "mnist_cnn_model.pth")
```

This file will be loaded by the Streamlit web application for predictions.

