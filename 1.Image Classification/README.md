# Image Classification — LeNet-5

Implementation of the original 1998 LeNet-5 architecture in **both PyTorch and Keras**, on MNIST.

## Why LeNet-5?
LeNet-5 is the architecture that proved deep convolutional networks work. Before it, handcrafted features dominated vision. Understanding it from scratch builds the mental model for every modern CNN.

## Architecture

```
Input: 28×28×1 (MNIST, zero-padded from original 32×32)

conv1:  Conv2d(1→6,  k=5, pad=2)  → Tanh  → 28×28×6
pool1:  AvgPool2d(2×2)             →         14×14×6
conv2:  Conv2d(6→16, k=5)         → Tanh  → 10×10×16
pool2:  AvgPool2d(2×2)             →         5×5×16
flatten: 400
fc1:    400 → 120 → Tanh
fc2:    120 → 84  → Tanh
fc3:    84  → 10  → Softmax
```

## Training

| Setting | Value |
|---------|-------|
| Optimiser | Adam |
| Learning rate | 0.001 |
| Batch size | 64 |
| Epochs | 10 |
| Loss | CrossEntropyLoss |

**Test accuracy: ~99.1%**

## Framework Comparison (PyTorch vs Keras)

The same architecture in two frameworks shows how different APIs express identical operations — useful for reading code written in either framework.

| | PyTorch | Keras |
|-|---------|-------|
| Model definition | Class inheriting `nn.Module` | `Sequential([...])` |
| Forward pass | Explicit `forward()` method | Implicit via `call()` |
| Training loop | Manual loss.backward() + optimizer.step() | `model.fit()` |
| Verbosity | More explicit, more control | More concise |

## Files
- `LeNet5+Pytorch.ipynb` — PyTorch implementation with manual training loop
- `LeNet5+with+MNIST+Keras.ipynb` — Keras implementation

**Reference:** [LeCun et al., 1998](https://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
