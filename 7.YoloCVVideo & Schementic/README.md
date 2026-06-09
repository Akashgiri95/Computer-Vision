# YOLO Video Analytics & U-Net Semantic Segmentation

Two notebooks: real-time object tracking on video, and pixel-wise image segmentation.

---

## Notebook 1: YOLO CV Lab — Video Object Tracking

Extends YOLOv8 detection to video streams by integrating a **ByteTrack** tracker.

### Why Tracking Is Hard
Detection alone gives bounding boxes per frame — but the IDs change each frame. Tracking assigns a **persistent ID** to each object across frames, enabling trajectory analysis, counting, and behaviour understanding.

### Pipeline

```
Video Frame
    ↓
YOLOv8 Detection → (boxes, class_ids, confidence_scores)
    ↓
ByteTrack → match detections to existing tracks (Hungarian algorithm + Kalman filter prediction)
    ↓
Annotate frame with IDs + trajectories
    ↓
Write output video
```

```python
tracker    = sv.ByteTrack()
box_ann    = sv.BoxAnnotator()
label_ann  = sv.LabelAnnotator()
trace_ann  = sv.TraceAnnotator()   # draws movement trails

def process_frame(frame, _):
    results  = model(frame, verbose=False)[0]
    detects  = sv.Detections.from_ultralytics(results)
    detects  = tracker.update_with_detections(detects)
    return box_ann.annotate(trace_ann.annotate(frame, detects), detects)
```

---

## Notebook 2: U-Net Semantic Segmentation

### The Problem with Classification for Segmentation
A classification model gives one label per image. Segmentation requires **one label per pixel** — each pixel belongs to a class. Standard CNNs with pooling lose spatial resolution.

### U-Net Architecture
```
Encoder (contracting path):
  64 → 128 → 256 → 512  (Conv+Conv+MaxPool at each level)

Bottleneck:
  512 → 1024

Decoder (expanding path):
  1024+512 → 512+256 → 256+128 → 128+64 → final Conv(1×1)
   ↑            ↑           ↑          ↑
  skip       skip         skip       skip    ← concatenated from encoder
```

**Why skip connections?** Pooling in the encoder destroys fine spatial detail. Skip connections forward the encoder's feature maps directly to the decoder, enabling precise localisation at full resolution.

**Loss:** Dice Loss
```python
dice_loss = 1 - (2 * TP) / (2*TP + FP + FN)
```
Better than BCE for segmentation because it directly optimises the overlap between prediction and ground truth — handles class imbalance naturally.
