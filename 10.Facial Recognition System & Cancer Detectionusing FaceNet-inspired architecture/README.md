# Facial Recognition & Cancer Detection

Two medical/biometric AI projects using deep CNNs: face verification with metric learning, and histopathology-based cancer detection.

---

## Project 1: Facial Recognition with Triplet Loss (FaceNet-inspired)

### The Approach
Standard classification (softmax over N people) doesn't generalise to new identities not seen during training. Metric learning solves this by learning an **embedding space** where faces of the same person are close and faces of different people are far apart.

### Triplet Loss

```
Triplet = (Anchor face, Positive face [same person], Negative face [different person])

L = max(||f(A) - f(P)||² - ||f(A) - f(N)||² + margin, 0)
```

The network is penalised whenever the anchor-to-negative distance is not at least `margin` larger than the anchor-to-positive distance. This forces the embedding space to be well-separated.

### Architecture

```python
class FaceNet(nn.Module):
    def __init__(self):
        # Backbone: Conv blocks → pooling → flatten
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)   # 160×160→160×160
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        # ...
        self.fc    = nn.Linear(512, 128)                # 128-dim embedding

    def forward(self, x):
        # Returns L2-normalised 128-dim embedding vector
        return F.normalize(self.fc(x), p=2, dim=1)
```

### Dataset
**CelebA Face Recognition Triplets** (`quadeer15sh/celeba-face-recognition-triplets`) — pre-organised as (anchor, positive, negative) triplets from CelebA faces.

### Inference
```python
emb_stored = model(enrolled_face)   # store embedding for known person
emb_query  = model(query_face)

similarity = F.cosine_similarity(emb_stored, emb_query)
is_match   = similarity > threshold  # tune threshold on validation set
```

---

## Project 2: Breast Cancer Detection (IDC Histopathology)

### The Problem
**Invasive Ductal Carcinoma (IDC)** is the most common form of breast cancer. Pathologists manually examine histopathology slides — a time-consuming, error-prone process. Automated detection from small 50×50 pixel patches can assist clinical decisions.

### Dataset
**Breast Histopathology Images** (`paultimothymooney/breast-histopathology-images`)
- 277,524 patches extracted from 162 whole-mount slides
- Each patch: 50×50×3 pixels, labelled 0 (no IDC) or 1 (IDC)
- Class distribution: ~60% negative, ~40% positive

### Architecture
```python
model = Sequential([
    Conv2D(32, 3, activation='relu') → BatchNorm → MaxPool,
    Conv2D(64, 3, activation='relu') → BatchNorm → MaxPool,
    Conv2D(128, 3, activation='relu') → BatchNorm → MaxPool,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu') → Dropout(0.5),
    Dense(1, activation='sigmoid')   # binary classification
])
```

### Handling Class Imbalance
```python
# Compute class weights to penalise misclassification of minority class
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight('balanced', classes=[0,1], y=y_train)
```

### Evaluation Metrics
Accuracy alone is misleading with class imbalance. Key metrics:
- **AUC-ROC** — discrimination ability across all thresholds
- **F1 Score** — balance of precision and recall
- **Confusion matrix** — understand false negative rate (missed cancers)

Medical AI requires minimising false negatives (missed cancers) even at cost of more false positives.
