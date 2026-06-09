# Image Acquisition & Face Detection Pipeline

Complete end-to-end pipeline from image capture to face detection, including preprocessing steps that significantly affect detection accuracy.

## Pipeline Steps

```
1. Image Acquisition
   ├── Webcam capture (cv2.VideoCapture)
   └── Static image loading (cv2.imread)
        ↓
2. Preprocessing
   ├── Resize to standard dimensions
   ├── Convert to grayscale (Haar cascade requires single-channel)
   └── Histogram equalisation (improves contrast in uneven lighting)
        ↓
3. Haar Cascade Detection
   ├── Multi-scale detection (scaleFactor pyramid)
   └── Confidence filtering (minNeighbors)
        ↓
4. Post-processing
   ├── Draw bounding boxes
   ├── Crop ROI for downstream use (recognition, expression analysis)
   └── Save or stream result
```

## Key Preprocessing: Histogram Equalisation

Poor lighting is the main cause of missed detections. CLAHE (Contrast Limited Adaptive Histogram Equalisation) dramatically improves detection in shadows and bright spots:

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray_eq = clahe.apply(gray)
```

**CLAHE vs global equalisation:** Applies equalisation locally to tiles rather than the full image — avoids over-amplifying noise in already-bright regions.

## Limitations of Haar Cascade
- Struggles with non-frontal faces (profile, tilted)
- Poor performance in low resolution (<24×24 pixels)
- Sensitive to heavy occlusion (masks, sunglasses)

For production use, replace with MTCNN or RetinaFace for better accuracy across poses and conditions.
