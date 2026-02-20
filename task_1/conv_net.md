# Convolutional Neural Networks (CNNs)

Convolutional Neural Networks (CNNs), also known as ConvNets, are a class of deep learning architectures inspired by the human visual system. They are specifically designed to process structured grid-like data, particularly images. CNNs are widely used in computer vision tasks such as image classification, object detection, and pattern recognition.

Unlike traditional fully connected neural networks, CNNs preserve the spatial relationships between pixels and automatically learn hierarchical features through convolution operations. Early layers detect simple patterns such as edges and textures, while deeper layers recognize more complex shapes and objects.

CNNs are computationally efficient because they process local regions of the image instead of connecting every pixel to every neuron. This reduces the number of learnable parameters and improves scalability.


![1111](https://github.com/user-attachments/assets/133ccb80-72a2-46ef-87fe-c31385a58927)


---

## Structure of a CNN Architecture

A complete CNN is a sequence of layers. Each layer transforms the input volume into another volume through a differentiable function.

For example, consider an input image of size:

32 × 32 × 3 (RGB image)

### 1. Input Layer

The input layer receives raw image data.

- Stores pixel values (Width × Height × Depth).
- Example: 32 × 32 × 3 for RGB images.
- Preserves spatial structure for further feature extraction.

---

### 2. Convolutional Layer

The convolutional layer is the core building block of a CNN.

It applies learnable filters (kernels) that slide over the image and copute dot products between filter weights and image patches.

Key concepts:
- Filters typically have size 3×3, 5×5, etc.
- Each filter produces one feature map.
- Multiple filters detect different features.

The convolution operation:
- Performs element-wise multiplication.
- Sums the results.
- Moves across the image using a defined stride.
- Produces feature maps that highlight edges, textures, and patterns.

Hyperparameters:
- Number of filters (controls depth);
- Stride (movement step size);
- Padding (valid, same, full).

---

### 3. Activation Layer

After convolution, an activation function introduces non-linearity.

Common activation functions:
- ReLU (Rectified Linear Unit)
- Tanh
- Leaky ReLU

ReLU is most commonly used:

f(x) = max(0, x)

The output dimensions remain unchanged (e.g., 32 × 32 × 12).

---

### 4. Pooling Layer

Pooling reduces spatial dimensions to:
- Decrease computation
- Reduce memory usage
- Prevent overfitting

Common types:
- Max Pooling
- Average Pooling

Example:
2 × 2 max pooling with stride 2 reduces:

32 × 32 × 12 → 16 × 16 × 12

Depth remains unchanged.

---

### 5. Flattening

Flattening converts multi-dimensional feature maps into a one-dimensional vector.

Example:
16 × 16 × 12 → 3072-dimensional vector

This vector is passed to fully connected layers.

---

### 6. Fully Connected Layer

The fully connected (dense) layer performs high-level reasoning.

Each neuron connects to all neurons from the previous layer and produces classification scores.

---

### 7. Output Layer

The output layer converts scores into probabilities.

- Sigmoid (binary classification)
- Softmax (multi-class classification)

---

## How Convolution Works

![222](https://github.com/user-attachments/assets/d3f12ee7-b168-4d05-9b40-288646c8e535)


The convolution operation:
1. A filter (kernel) slides across the image.
2. Element-wise multiplication is performed.
3. Values are summed.
4. A feature map is generated.
5. Multiple filters capture different features.

This process preserves spatial relationships while reducing parameters compared to fully connected networks.

---

## CNN vs Other Neural Networks

Neural networks consist of:
- Input layer
- Hidden layers
- Output layer

While feedforward networks connect all neurons fully, CNNs specialize in spatial data.

Compared to:
- Recurrent Neural Networks (RNNs) → used for sequential data (text, speech)
- CNNs → used for images and structured grid data

CNNs replaced manual feature extraction methods and now automatically learn features through backpropagation and gradient descent. However, they can be computationally intensive and often require GPUs for efficient training.

---

## Applications of CNNs

CNNs power modern computer vision systems:

- Healthcare (tumor detection in radiology)
- Automotive (lane detection)
- Retail (visual search)
- Marketing (face tagging in social media)

CNNs have become foundational models in artificial intelligence for visual data processing.

---

# Practical Cybersecurity Example: DDoS Detection with a 1D CNN (Time-Series)

## Problem setup
A common sign of a DDoS attack is a sudden and sustained increase in request rate (requests per second / per minute).
We can frame this as a **binary classification** task on short time windows:
- **0 = normal traffic**
- **1 = DDoS traffic**

Instead of using images, we treat the request-rate sequence as a **1D signal**, which is well-suited to **1D convolution**.
A 1D CNN can learn local temporal patterns such as spikes, bursts, and sustained high-rate periods.

## Data (included in this report)
The dataset is **generated synthetically** in code:
- Normal windows: baseline traffic with small random noise
- Attack windows: elevated baseline + bursts/spikes

This mirrors real traffic behavior while avoiding external files.

## Visualizations produced
The code below generates:
1. **Traffic time-series plot** (normal vs attack example)
2. **Training loss curve**
3. **Confusion matrix** on the test set

---

## Python code 

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------
# 1) Synthetic dataset generator
# -----------------------------
def make_windows(n_samples=2000, window_size=60, attack_ratio=0.5, seed=7):
    """
    Returns:
      X: shape (n_samples, window_size) request-rate windows
      y: shape (n_samples,) labels {0=normal, 1=attack}
    """
    rng = np.random.default_rng(seed)
    n_attack = int(n_samples * attack_ratio)
    n_normal = n_samples - n_attack

    normal_base = rng.uniform(5, 15, size=(n_normal, 1))
    normal = normal_base + rng.normal(0, 2, size=(n_normal, window_size))
    normal = np.clip(normal, 0, None)

    attack_base = rng.uniform(30, 60, size=(n_attack, 1))
    attack = attack_base + rng.normal(0, 5, size=(n_attack, window_size))

    for i in range(n_attack):
        n_bursts = rng.integers(1, 4)
        for _ in range(n_bursts):
            pos = rng.integers(0, window_size)
            width = rng.integers(2, 8)
            amp = rng.uniform(20, 60)
            attack[i, pos:pos+width] += amp

    attack = np.clip(attack, 0, None)

    X = np.vstack([normal, attack]).astype(np.float32)
    y = np.array([0]*n_normal + [1]*n_attack, dtype=np.int64)

    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


# -----------------------------
# 2) Train/test split + loaders
# -----------------------------
X, y = make_windows(n_samples=2500, window_size=60, attack_ratio=0.5, seed=7)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

mu = X_train.mean()
sigma = X_train.std() + 1e-6
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

X_train_t = torch.tensor(X_train).unsqueeze(1)  # (N, 1, 60)
X_test_t  = torch.tensor(X_test).unsqueeze(1)
y_train_t = torch.tensor(y_train)
y_test_t  = torch.tensor(y_test)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=256, shuffle=False)


# -----------------------------
# 3) 1D CNN model
# -----------------------------
class DDoS1DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 60 -> 30

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 30 -> 15
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 15, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 2 classes: normal/attack
        )

    def forward(self, x):
        x = self.net(x)
        return self.classifier(x)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = DDoS1DCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# -----------------------------
# 4) Training loop
# -----------------------------
def train_epoch():
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
    return total_loss / len(train_loader.dataset)

def evaluate():
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            p = torch.argmax(logits, dim=1).cpu().numpy()
            preds.append(p)
            trues.append(yb.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    acc = (preds == trues).mean()
    return acc, preds, trues

losses = []
epochs = 8
for ep in range(1, epochs+1):
    l = train_epoch()
    losses.append(l)
    acc, _, _ = evaluate()
    print(f"Epoch {ep}/{epochs} - loss={l:.4f} - test_acc={acc:.4f}")


# -----------------------------
# 5) Visualizations (save PNGs)
# -----------------------------

# 5.1 Traffic example plot (one normal vs one attack window, UN-normalized for readability)
X_raw, y_raw = make_windows(n_samples=200, window_size=60, attack_ratio=0.5, seed=11)
normal_ex = X_raw[y_raw == 0][0]
attack_ex = X_raw[y_raw == 1][0]

plt.figure()
plt.plot(normal_ex, label="Normal window")
plt.plot(attack_ex, label="Attack window")
plt.title("Synthetic traffic windows (requests over time)")
plt.xlabel("Time step")
plt.ylabel("Requests (approx.)")
plt.legend()
plt.tight_layout()
plt.savefig("traffic_windows.png", dpi=200)
plt.show()

# 5.2 Loss curve
plt.figure()
plt.plot(losses)
plt.title("Training loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=200)
plt.show()

# 5.3 Confusion matrix (no sklearn needed)
acc, preds, trues = evaluate()
cm = np.zeros((2, 2), dtype=int)
for t, p in zip(trues, preds):
    cm[t, p] += 1

plt.figure()
plt.imshow(cm, aspect="auto")
plt.title("Confusion matrix (test set)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks([0, 1], ["Normal", "Attack"])
plt.yticks([0, 1], ["Normal", "Attack"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
plt.show()

print("Saved: traffic_windows.png, loss_curve.png, confusion_matrix.png")



---

<img width="1280" height="960" alt="confusion_matrix" src="https://github.com/user-attachments/assets/775d7a92-a579-452f-9b59-814a1c1e4b92" />

<img width="1280" height="960" alt="loss_curve" src="https://github.com/user-attachments/assets/94c37f75-cac2-44d1-8ef3-9086d6f6a6f5" />

<img width="1280" height="960" alt="traffic_windows" src="https://github.com/user-attachments/assets/0f739c13-c37b-46c8-a92d-810a6efc612c" />

