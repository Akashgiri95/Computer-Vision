# Image Classification — CIFAR-10 & Transfer Learning

Two notebooks progressing from training a custom CNN to fine-tuning pre-trained ImageNet models.

---

## Notebook 1: CIFAR-10 with Custom CNN

### The Problem
CIFAR-10: 60,000 colour images (32×32×3) across 10 classes (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck). Much harder than MNIST — colour, texture, background clutter.

### Architecture

```
Input: 32×32×3

Block 1: Conv(32,3×3) → BN → ReLU → Conv(32,3×3) → BN → ReLU → MaxPool(2) → Dropout(0.25)
Block 2: Conv(64,3×3) → BN → ReLU → Conv(64,3×3) → BN → ReLU → MaxPool(2) → Dropout(0.25)
Flatten → Dense(512) → BN → ReLU → Dropout(0.5) → Dense(10, softmax)
```

**Why Batch Normalisation?** Stabilises training by normalising activations per mini-batch — allows higher learning rates and faster convergence without careful weight initialisation.

**Why Dropout?** Randomly zeroes 25–50% of neurons during training, preventing co-adaptation. The network learns redundant representations.

### Results

| Epochs | Test Accuracy | Notes |
|--------|--------------|-------|
| 10 | ~72% | Underfitting — needs more epochs |
| 20 | ~78–82% | Good convergence |

---

## Notebook 2: Transfer Learning with VGG16

### The Problem
Cats vs Dogs dataset (~25,000 images). Training from scratch on small datasets overfits quickly. Transfer learning reuses features learned from 1.2M ImageNet images.

### Two-Phase Training Strategy

**Phase 1 — Feature Extraction** (freeze backbone)
```python
base_model = VGG16(weights='imagenet', include_top=False)
base_model.trainable = False   # freeze all VGG16 weights

model = Sequential([base_model, GlobalAvgPool2D(), Dense(512), Dense(1, sigmoid)])
# Train only top layers for 10 epochs at lr=0.0001
```

**Phase 2 — Fine-Tuning** (unfreeze last conv blocks)
```python
base_model.trainable = True
# Re-freeze early layers (low-level features are universal)
for layer in base_model.layers[:-4]:
    layer.trainable = False
# Continue training end-to-end at lr=0.00001
```

### Why This Works
Early VGG16 layers detect universal features (edges, textures). Later layers detect ImageNet-specific patterns (dog ears, bird beaks). We keep the universal features and adapt the task-specific ones.

| Approach | Validation Accuracy |
|----------|-------------------|
| Training from scratch | ~60% |
| Transfer learning (Phase 1 only) | ~90% |
| Transfer + fine-tuning (Phase 2) | ~95% |
