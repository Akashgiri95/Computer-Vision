# Object Detection — YOLOv11

Running inference with the latest YOLO generation (v11) on real-world images.

## YOLO Architecture (How it Works)
Unlike two-stage detectors (R-CNN) that first propose regions then classify, YOLO frames detection as a single regression problem — the network predicts bounding boxes and class probabilities directly from the full image in one forward pass.

```
Input Image → Backbone (feature extraction) → Neck (FPN multi-scale) → Head (predict boxes + classes)
```

## Models Used

| Model | Parameters | Speed | mAP(50-95) |
|-------|-----------|-------|------------|
| YOLOv11n (nano) | 2.6M | ~4ms/img | 39.5 |
| YOLOv11s (small) | 9.4M | ~8ms/img | 47.0 |

**Rule of thumb:** Nano for real-time on CPU; Small when you can afford a small GPU.

## What Was Done

```python
from ultralytics import YOLO

model   = YOLO('yolo11n.pt')               # load pre-trained nano
results = model(['Gathering.jpg', 'roadcross.jpg'], conf=0.4)

for r in results:
    r.show()                               # visualise bounding boxes
    print(r.boxes.cls, r.boxes.conf)       # class IDs and confidence scores
```

**Test images:**
- `Gathering.jpg` — crowd scene; detects persons, bags
- `roadcross.jpg` — road crossing; detects cars, persons, traffic lights

## Key Concepts

**Confidence threshold:** `conf=0.4` means only predictions with ≥40% confidence are shown. Too low → many false positives. Too high → misses distant/occluded objects.

**NMS (Non-Maximum Suppression):** When multiple boxes overlap for the same object, NMS keeps only the highest-confidence box (based on IoU threshold, default 0.45).

**Anchor-free (YOLOv8+):** Modern YOLO versions predict box centre + size directly rather than offsets from predefined anchor shapes — simpler and more accurate.
