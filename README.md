# Computer Vision

A structured progression through computer vision — from building convolution filters by hand to deploying real-time traffic monitoring and medical AI systems.

Each folder is a standalone deep-dive: the notebook explains the theory, implements from scratch, and demonstrates on real data.

---

## Curriculum

| # | Topic | What was built | Key result |
|---|-------|----------------|------------|
| 1 | [CNN Fundamentals](#1-cnn-fundamentals) | Custom kernels, padding/stride experiments | Understood spatial dimension formula; effect on accuracy |
| 2 | [Image Classification](#2-image-classification) | LeNet-5, CIFAR-10 CNN, Transfer Learning (VGG16) | MNIST ~99% · CIFAR-10 ~80% · Cats & Dogs fine-tuned |
| 3 | [Object Detection](#3-object-detection) | YOLOv11 inference, Fast R-CNN (ResNet50 backbone) | Real-time bounding boxes on crowd/road images |
| 4 | [Semantic Segmentation](#4-semantic-segmentation) | U-Net encoder-decoder from scratch | Pixel-wise segmentation with skip connections |
| 5 | [Feature Descriptors](#5-feature-descriptors) | SIFT, SURF, ORB matching pipeline | Cross-image keypoint matching with Lowe's ratio test |
| 6 | [Corner Detection](#6-corner-detection) | Harris + FAST implemented from scratch in NumPy | Corner response maps on chessboard images |
| 7 | [Face Detection & Recognition](#7-face-detection--recognition) | Haar Cascade + FaceNet-style CNN with triplet loss | CelebA face verification |
| 8 | [Video Analytics](#8-video-analytics) | Traffic monitoring: speed, red-light violation, congestion | YOLOv8 on 3 real traffic clips |
| 9 | [Medical AI](#9-medical-ai) | Cancer detection from histopathology slides | IDC breast cancer dataset |
| 10 | [OCR — Xerox Case Study](#10-ocr--xerox-case-study) | Connected component analysis + shape descriptors + EasyOCR | Why JBIG2 compression breaks OCR |
| 11 | [Advanced Labs](#11-advanced-labs) | Image stitching (DL + classical), full facial recognition system | Panoramic stitching, TESLA vision pipeline |

---

## 1. CNN Fundamentals

**Folders:** `3.CNN/` · `9. Impact of Padding & Stride/`

### Custom Filters (`Building Your Custom Filters`)
Built convolution kernels manually in NumPy — Sobel edge detection, Gaussian blur, sharpening. Visualised what each filter extracts from an image before using `nn.Conv2d`. This builds the intuition for why deeper networks compose many simple operations.

### Padding & Stride Experiments (`Padding & Strides in CNN`)
Ran controlled experiments comparing `valid` vs `same` padding and strides of 1 vs 2 across 4 CNN configurations on MNIST. Measured accuracy and training time for each.

```
Output size = ⌊(Input + 2×Padding - Kernel) / Stride⌋ + 1
```

| Config | Padding | Stride | Test Accuracy | Notes |
|--------|---------|--------|---------------|-------|
| Baseline | `same` | 1 | ~99.1% | Full spatial resolution retained |
| No padding | `valid` | 1 | ~98.8% | Slight loss at edges |
| Same + stride 2 | `same` | 2 | ~98.4% | 2× faster, minor accuracy drop |
| Valid + stride 2 | `valid` | 2 | ~98.0% | Most aggressive downsampling |

### Fast R-CNN (Region-Based Object Detection)
Simplified Faster R-CNN using a frozen ResNet50 backbone: selective search → RoI pooling → two-headed output (class logits + box regression). Demonstrates why two-stage detectors are more accurate but slower than YOLO.

---

## 2. Image Classification

**Folders:** `1.Image Classification/` · `1.Implementing Basic Image Classification with CNNs/`

### LeNet-5 from Scratch (PyTorch + Keras)
Implemented the original 1998 LeNet-5 architecture in both PyTorch and Keras/TensorFlow on MNIST. The comparison shows how different frameworks express identical architectures.

```python
# LeNet-5 architecture (PyTorch)
conv1: Conv2d(1→6, k=5, pad=2)  → 28×28×6
pool1: AvgPool(2×2)              → 14×14×6
conv2: Conv2d(6→16, k=5)        → 10×10×16
pool2: AvgPool(2×2)              → 5×5×16
fc1:   120  →  fc2: 84  →  fc3: 10
```

Trained for 10 epochs, Adam lr=0.001 — **Test accuracy: ~99.1%**

### CIFAR-10 CNN
Custom CNN on 60,000 colour images across 10 classes. Architecture uses double conv blocks with batch normalisation and dropout.

```
Block 1: Conv(32) → BN → Conv(32) → BN → MaxPool → Dropout(0.25)
Block 2: Conv(64) → BN → Conv(64) → BN → MaxPool → Dropout(0.25)
Dense: 512 → Dropout(0.5) → 10 (softmax)
```
Trained 20 epochs — **Test accuracy: ~78–82%**

### Transfer Learning — VGG16 on Cats vs Dogs
Fine-tuned pre-trained VGG16 (ImageNet weights) on the Dogs vs Cats dataset with two-phase training:
- **Phase 1:** Freeze backbone, train only top dense layers (10 epochs)
- **Phase 2:** Unfreeze last conv blocks, fine-tune end-to-end at lr=0.0001

**Final validation accuracy: ~95%** — vs ~60% training from scratch

---

## 3. Object Detection

**Folder:** `4.Object Detection/`

### YOLOv11 Inference
Ran YOLOv11 nano and small pre-trained models on crowd gathering and road-crossing images. Compared nano (speed) vs small (accuracy) at different confidence thresholds.

```python
model = YOLO('yolo11n.pt')  # or yolo11s.pt
results = model('roadcross.jpg', conf=0.4)
```

**Nano:** ~4ms/image · **Small:** ~8ms/image · Small gains ~4% mAP over nano

---

## 4. Semantic Segmentation

**Folder:** `7.YoloCVVideo & Schementic/`

### U-Net Encoder-Decoder
Full U-Net implementation with skip connections for pixel-wise segmentation. The skip connections (concatenating encoder feature maps to decoder) preserve fine spatial details that pure encoder-decoder networks lose through pooling.

```
Encoder: 3→64→128→256→512 (with MaxPool)
Bottleneck: 512→1024
Decoder: 1024→512→256→128→64 (with transposed conv + skip concat)
Output: 1×1 Conv → sigmoid
```

Loss: **Dice loss** (better than BCE for imbalanced foreground/background)

---

## 5. Feature Descriptors

**Folder:** `5. Feature Descriptors - SIFT, SURF, and ORB/`

Detected and matched keypoints between two images using all three descriptors. Each descriptor represents a different trade-off:

| Descriptor | Invariance | Descriptor Size | Speed | Patent |
|------------|-----------|----------------|-------|--------|
| SIFT | Scale + Rotation | 128-dim float | Slow | Free (expired) |
| SURF | Scale + Rotation | 64-dim float | Medium | Yes |
| ORB | Rotation | 256-bit binary | Fast | No |

Applied **Lowe's ratio test** (`d1/d2 < 0.75`) to filter false matches. ORB with BFMatcher Hamming distance gave real-time matching at comparable quality.

---

## 6. Corner Detection

**Folder:** `6.Corner Detector/`

Both detectors implemented **from scratch in NumPy** — not using `cv2.cornerHarris()`.

### Harris Corner Detector
```python
# Structure tensor from image gradients
Ix, Iy = sobel(gray, axis=1), sobel(gray, axis=0)
M = [[Ix², IxIy], [IxIy, Iy²]]  # averaged over local window
R = det(M) - k × trace(M)²      # corner response (k=0.04)
```
Corners: R >> 0 · Edges: R < 0 · Flat: R ≈ 0

### FAST Corner Detector
Pixel-ring test: compare 16 surrounding pixels to centre intensity. If ≥ N consecutive pixels are all brighter or all darker by `threshold`, it's a corner. Much faster than Harris — basis of ORB.

---

## 7. Face Detection & Recognition

**Folders:** `Face Detection(Haar Cascade)/` · `10.Facial Recognition System.../`

### Haar Cascade Face Detection
Real-time face detection pipeline: webcam capture → grayscale → Haar Cascade (`haarcascade_frontalface_default.xml`) → bounding box overlay. Separate backend/frontend scripts for server-side and display.

### Facial Recognition with Triplet Loss (CelebA)
FaceNet-inspired CNN trained with **triplet loss** on the CelebA triplets dataset:

```
L = max(||f(anchor) - f(positive)||² - ||f(anchor) - f(negative)||² + margin, 0)
```

The network learns a 128-dim embedding space where faces of the same person cluster together. At inference: cosine similarity between stored embeddings and query face.

**Dataset:** `quadeer15sh/celeba-face-recognition-triplets` (Kaggle)

---

## 8. Video Analytics

**Folder:** `8.Traffic Monitoring & Video Analytic.ipynb/`

Real-time traffic monitoring system built on YOLOv8 + supervision, tested on 3 actual traffic video clips.

**Features implemented:**
- Vehicle detection and classification (car, truck, bus, motorbike)
- Multi-object tracking (ByteTrack)
- **Speed estimation** (pixels/frame → km/h via calibration)
- **Red light violation detection** (crossing defined stop line on red)
- **Congestion detection** (vehicle density threshold per zone)
- Heatmap generation over time

```python
tracker = sv.ByteTrack()
annotator = sv.BoxAnnotator()
line_zone = sv.LineZone(start=Point(x1,y1), end=Point(x2,y2))
```

---

## 9. Medical AI

**Folder:** `10.Facial Recognition System & Cancer Detection.../`

### Breast Cancer Detection (IDC Histopathology)
Binary classification of 50×50 pixel histopathology patches: **IDC positive** (invasive ductal carcinoma) vs **IDC negative**.

- **Dataset:** `paultimothymooney/breast-histopathology-images` — 277,524 patches from 162 patients
- **Challenge:** Class imbalance (~60% negative, ~40% positive)
- **Architecture:** Custom CNN with ImageDataGenerator augmentation
- **Metrics:** Accuracy, AUC-ROC, precision, recall, F1 — confusion matrix analysis

---

## 10. OCR — Xerox Case Study

**Folder:** `2.OCR-Case Xerox/`

Investigation of the famous **Xerox JBIG2 compression bug** where scanned documents had digits silently swapped during copying. Explores why this escaped human detection but breaks OCR.

**Pipeline:**
1. Load document at three quality levels (original, lossless, lossy)
2. Connected component analysis — extract character blobs
3. Shape descriptors: **Hu Moments** (rotation-invariant) + aspect ratio
4. Distance-based matching to detect compression-introduced character substitutions
5. EasyOCR extraction + accuracy comparison across compression levels

**Key insight:** JBIG2 uses pattern substitution for compression — two visually similar glyphs (6 and 8) get mapped to the same pattern, silently changing document content. OCR fails because the bit-level representation is wrong, even though it looks correct to a human.

---

## 11. Advanced Labs

**Folder:** `11.LAB/`

### Image Stitching — Classical vs Deep Learning

| Method | Approach | Strength |
|--------|----------|----------|
| Classical (`Manual Image_Stitching_OpenCV`) | SIFT keypoints → RANSAC homography → perspective warp | Interpretable; works without training data |
| Deep Learning (`ImageStitching_DeepLearning`) | CNN estimates homography directly from image pairs | Handles textureless / low-feature regions better |

### Complete Facial Recognition System
Production-style pipeline: MTCNN face detection → alignment → FaceNet embedding → cosine similarity verification → identity lookup.

### TESLA Vision Pipeline
System design document for an autonomous vehicle perception pipeline — object detection, depth estimation, lane detection integration.

---

## Tech Stack

```
Deep Learning    PyTorch · TensorFlow / Keras
Object Detection Ultralytics YOLOv8 / YOLOv11 · supervision
Computer Vision  OpenCV · scikit-image · Pillow
OCR              EasyOCR · Tesseract
Data & Viz       NumPy · Pandas · Matplotlib · Seaborn
Platform         Jupyter Notebook · Google Colab · Kaggle
```

## Setup

```bash
pip install torch torchvision tensorflow opencv-python ultralytics \
            easyocr supervision scikit-image matplotlib pandas seaborn
```

> Training notebooks require a GPU — use Google Colab (free T4) for large datasets.
