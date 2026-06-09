# Computer Vision

A structured progression through computer vision — from CNN fundamentals to real-world applications including object detection, semantic segmentation, face recognition, medical AI, and traffic analytics.

---

## Table of Contents

| # | Topic | Notebooks | Key Libraries |
|---|-------|-----------|---------------|
| 1 | [CNN Fundamentals](#1-cnn-fundamentals) | Custom filters, padding & strides | PyTorch, NumPy |
| 2 | [Image Classification](#2-image-classification) | LeNet-5, CIFAR-10, Transfer Learning | PyTorch, TensorFlow/Keras |
| 3 | [Object Detection](#3-object-detection) | YOLOv11, Fast R-CNN | Ultralytics, PyTorch |
| 4 | [Semantic Segmentation](#4-semantic-segmentation) | U-Net from scratch | PyTorch, Kaggle |
| 5 | [Feature Descriptors](#5-feature-descriptors) | SIFT, SURF, ORB | OpenCV |
| 6 | [Corner Detection](#6-corner-detection) | Harris, FAST | OpenCV, SciPy |
| 7 | [Face Detection & Recognition](#7-face-detection--recognition) | Haar Cascade, FaceNet-style CNN | OpenCV, DeepFace |
| 8 | [Video Analytics](#8-video-analytics) | Traffic monitoring, real-time tracking | YOLOv8, supervision |
| 9 | [Medical AI](#9-medical-ai) | Cancer detection from histology | PyTorch |
| 10 | [OCR Case Study](#10-ocr---xerox-case-study) | Document digitisation pipeline | EasyOCR, Tesseract |
| 11 | [Advanced Labs](#11-advanced-labs) | Image stitching, full facial recognition | PyTorch, OpenCV |

---

## 1. CNN Fundamentals

**Folder:** `3.CNN/` · `9. Impact of Padding & Stride/`

| Notebook | Description |
|----------|-------------|
| `Building Your Custom Filters` | Build convolution kernels by hand — edge detection, sharpening, blur; visualise feature maps |
| `Padding & Strides in CNN` | `valid` vs `same` padding, stride effect on spatial dimensions; formulae + visual proofs |
| `Fast R-CNN (Region-based OD)` | Region proposals → RoI pooling → classification; simplified Faster R-CNN pipeline |

**Concepts:** Convolution, pooling, receptive field, feature maps, padding modes, stride

---

## 2. Image Classification

**Folder:** `1.Image Classification/` · `1.Implementing Basic Image Classification with CNNs/`

| Notebook | Description |
|----------|-------------|
| `LeNet-5 + PyTorch` | Implements the original 1998 LeNet-5 architecture on MNIST from scratch in PyTorch |
| `LeNet-5 + Keras` | Same architecture in Keras/TensorFlow — compare framework verbosity |
| `CIFAR-10 CNN` | Custom CNN for 10-class colour image classification — data augmentation, batch normalisation |
| `Transfer Learning` | Fine-tune pre-trained ImageNet models (VGG, ResNet) on custom datasets |

**Concepts:** LeNet-5, AlexNet-style training, transfer learning, data augmentation, dropout

---

## 3. Object Detection

**Folder:** `4.Object Detection/` · `9. Impact of Padding & Stride/Region based Object Detection/`

| Notebook | Description |
|----------|-------------|
| `YOLOv11 Inference` | Run YOLOv11n/s pre-trained weights on real images; visualise bounding boxes + confidence |
| `Fast R-CNN` | Understand region proposal networks, RoI pooling, two-stage detection pipeline |

**Concepts:** YOLO architecture, anchor boxes, NMS, IoU, two-stage vs one-stage detectors

---

## 4. Semantic Segmentation

**Folder:** `7.YoloCVVideo & Schementic/`

| Notebook | Description |
|----------|-------------|
| `U-Net Segmentation` | Full U-Net implementation — encoder-decoder with skip connections, Kaggle dataset |

**Concepts:** Encoder-decoder, skip connections, pixel-wise classification, Dice loss

---

## 5. Feature Descriptors

**Folder:** `5. Feature Descriptors - SIFT, SURF, and ORB/`

| Notebook | Description |
|----------|-------------|
| `SIFT, SURF, ORB` | Extract keypoints, compute descriptors, match features between images using BFMatcher |

**Concepts:** Scale-space, keypoint detection, descriptor matching, Lowe's ratio test

---

## 6. Corner Detection

**Folder:** `6.Corner Detector/`

| Notebook | Description |
|----------|-------------|
| `Harris & FAST` | Harris corner score, FAST (Features from Accelerated Segment Test), corner response maps |

**Concepts:** Structure tensor, corner response function, non-maximum suppression

---

## 7. Face Detection & Recognition

**Folder:** `Face Detection(Haar Cascade)/` · `Image Acquisition Face Detection(Haar)/` · `10.Facial Recognition System & Cancer Detection/`

| Notebook / Script | Description |
|-------------------|-------------|
| `face_detection_backend.py` | Flask/FastAPI backend serving Haar Cascade face detection |
| `face_detection_frontend.py` | Real-time webcam face detection frontend |
| `Image Acquisition Face Detection` | Full pipeline: image capture → preprocessing → Haar detection |
| `Facial Recognition CNN` | CNN trained with triplet loss for face verification (FaceNet-inspired) |

**Concepts:** Haar Cascade, Viola-Jones, embedding space, triplet loss, Siamese networks

---

## 8. Video Analytics

**Folder:** `7.YoloCVVideo & Schementic/` · `8.Traffic Monitoring & Video Analytic.ipynb/`

| Notebook | Description |
|----------|-------------|
| `YOLO CV Lab` | YOLOv8 on video streams — object tracking, trajectory visualisation |
| `Traffic Monitoring` | Vehicle detection + counting on 3 traffic video clips using YOLOv8 + supervision |
| `Video Analytics` | Extended pipeline: speed estimation, zone analytics, heatmaps |

**Concepts:** Multi-object tracking (ByteTrack), DeepSORT, counting lines, zone detection

---

## 9. Medical AI

**Folder:** `10.Facial Recognition System & Cancer Detectionusing FaceNet-inspired architecture/`

| Notebook | Description |
|----------|-------------|
| `Cancer Detection` | Histopathology image classification — CNN trained on cell imagery to distinguish malignant vs benign |

**Concepts:** Medical image classification, class imbalance, confusion matrix, ROC-AUC

---

## 10. OCR — Xerox Case Study

**Folder:** `2.OCR-Case Xerox/`

| Notebook | Description |
|----------|-------------|
| `OCR Case Xerox` | End-to-end document digitisation: image quality analysis (lossless vs lossy), EasyOCR extraction, post-processing |

**Concepts:** EasyOCR, Tesseract, image compression artefacts, document preprocessing

---

## 11. Advanced Labs

**Folder:** `11.LAB/`

| Notebook | Description |
|----------|-------------|
| `Facial Recognition System` | Complete production-style system: detection → alignment → embedding → verification |
| `Image Stitching (Deep Learning)` | Homography estimation with CNNs for panoramic image creation |
| `Image Stitching (OpenCV)` | Classical SIFT + RANSAC homography + warping pipeline |
| `Lab 5` | Mixed practicals |

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Deep Learning | PyTorch, TensorFlow, Keras |
| Computer Vision | OpenCV, scikit-image, Pillow |
| Object Detection | Ultralytics (YOLOv8, YOLOv11), supervision |
| OCR | EasyOCR, Tesseract |
| Data & Viz | NumPy, Pandas, Matplotlib, Seaborn |
| Environment | Jupyter Notebook, Google Colab |

## Setup

```bash
pip install torch torchvision opencv-python ultralytics easyocr \
            matplotlib numpy pandas scikit-image supervision
```

> Some notebooks require a GPU (Colab recommended for training notebooks).
