# Face Detection — Haar Cascade (Real-time)

Real-time face detection system using OpenCV's Haar Cascade classifier, split into a backend server and frontend display.

## How Haar Cascade Works

The Viola-Jones algorithm (2001) uses:
1. **Haar features:** Simple rectangular filters that compute intensity differences (similar to Sobel/Laplacian at different scales)
2. **Integral images:** Pre-computed cumulative sum allowing any rectangular feature to be evaluated in O(1) time regardless of rectangle size
3. **AdaBoost:** Selects the most discriminative features from 160,000+ candidates; combines weak classifiers into a strong one
4. **Cascade structure:** 38 stages — early stages quickly reject non-faces, later stages apply more expensive checks only to candidates that passed

**Result:** >95% detection rate at <1ms per frame on CPU.

## Files

| File | Description |
|------|-------------|
| `face_detection_backend.py` | Server exposing `/detect` endpoint — receives image, returns detected face bounding boxes as JSON |
| `face_detection_frontend.py` | Captures webcam frames, sends to backend, draws results overlay |
| `Feature Decorators.ipynb` | Feature extraction utilities and visualisation |

## Detection Pipeline
```python
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_faces(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,    # image pyramid: each level 10% smaller
        minNeighbors=5,     # how many nearby detections to confirm a face
        minSize=(30, 30)    # minimum face size in pixels
    )
    return faces  # list of (x, y, w, h) rectangles
```

**Key parameters:**
- `scaleFactor` — lower → more detections, slower; typical: 1.05–1.3
- `minNeighbors` — higher → fewer false positives, may miss real faces; typical: 3–6
