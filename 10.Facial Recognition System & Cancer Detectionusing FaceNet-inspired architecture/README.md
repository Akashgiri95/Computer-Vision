# Facial Recognition & Cancer Detection (FaceNet-inspired)

| Notebook | Description |
|----------|-------------|
| `Facial_Recog_CNN.ipynb` | FaceNet-inspired CNN using triplet loss for face verification — embedding space training |
| `Cancer_Detection.ipynb` | Histopathology slide classification — malignant vs benign cell detection |

## Facial Recognition
- **Architecture:** Shared-weight CNN → 128-dim embedding space
- **Loss:** Triplet loss — `L = max(d(a,p) − d(a,n) + margin, 0)`
- **Inference:** Cosine similarity between stored embeddings and query face

## Cancer Detection
- **Task:** Binary classification on histopathology images
- **Metrics:** Accuracy, AUC-ROC, confusion matrix, F1-score
- **Challenge:** Class imbalance handling with weighted loss / oversampling
