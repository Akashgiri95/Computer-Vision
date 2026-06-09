# Padding, Strides & Region-Based Object Detection

Two experiments that build deep understanding of spatial dimensions in CNNs and the two-stage detection paradigm.

---

## Lab 1: Padding & Strides in Convolutional Layers

### The Question
How does padding and stride choice affect the feature map size, training accuracy, and speed?

### Experiment
Trained 4 identical CNN architectures on MNIST, varying only padding and stride. Each trained for 10 epochs; measured final test accuracy and training time.

```python
experiments = [
    {"name": "Baseline (same, stride 1)",   "padding": "same",  "strides": (1,1)},
    {"name": "Valid padding, stride 1",      "padding": "valid", "strides": (1,1)},
    {"name": "Same padding, stride 2",       "padding": "same",  "strides": (2,2)},
    {"name": "Valid padding, stride 2",      "padding": "valid", "strides": (2,2)},
]
```

### Spatial Dimension Formula
```
Output = ⌊(Input + 2×Padding - Kernel) / Stride⌋ + 1

Example: Input=28, Kernel=3, Padding=1 (same), Stride=1
→ ⌊(28 + 2 - 3) / 1⌋ + 1 = 28   ✓ (size preserved)

Example: Input=28, Kernel=3, Padding=0 (valid), Stride=2
→ ⌊(28 + 0 - 3) / 2⌋ + 1 = 13
```

### Results

| Config | Final Accuracy | Training Time |
|--------|---------------|---------------|
| Same padding, stride 1 | ~99.1% | Baseline |
| Valid padding, stride 1 | ~98.8% | Similar |
| Same padding, stride 2 | ~98.4% | ~40% faster |
| Valid padding, stride 2 | ~98.0% | ~45% faster |

**Takeaway:** Stride-2 convolutions are a common alternative to MaxPooling — they downsample while learning the optimal subsampling, at a small accuracy cost.

---

## Lab 2: Simplified Faster R-CNN

### Two-Stage vs One-Stage Detection
YOLO (one-stage): predict boxes and classes simultaneously — fast but less accurate on small objects.
R-CNN (two-stage): first generate region proposals, then classify each region — slower but more accurate.

### Pipeline Implemented
```
Input Image
    ↓
ResNet50 Backbone (frozen early layers, trainable late layers)
    ↓
Feature Map: spatial feature representation of full image
    ↓
Region Proposals: selective search generates ~2000 candidate boxes
    ↓
RoI Pooling: extract fixed-size (7×7) features for each proposal
    ↓
Classification Head → class probabilities (21 classes)
Box Regression Head → refined box coordinates (Δx, Δy, Δw, Δh)
```

### Why ResNet50 Backbone?
Skip connections in ResNet prevent vanishing gradients in deep networks. The backbone is pre-trained on ImageNet — it extracts rich semantic features without needing to be retrained from scratch.

```python
backbone = keras.applications.ResNet50(
    include_top=False, weights='imagenet',
    input_shape=(None, None, 3)   # variable input size
)
for layer in backbone.layers[:100]:
    layer.trainable = False   # freeze early layers
```
