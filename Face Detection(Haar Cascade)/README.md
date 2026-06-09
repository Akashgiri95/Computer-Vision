# Face Detection — Haar Cascade

| File | Description |
|------|-------------|
| `face_detection_backend.py` | Backend server exposing face detection endpoint |
| `face_detection_frontend.py` | Real-time webcam capture + detection display |
| `Feature Decorators.ipynb` | Feature extraction and decoration utilities |

**Pipeline:** Webcam frame → grayscale conversion → Haar Cascade detector → bounding box overlay → display

**Key learning:** Viola-Jones algorithm; integral images for fast feature computation; cascade of classifiers for real-time speed.
