# Traffic Monitoring & Video Analytics

End-to-end intelligent traffic system: vehicle detection, multi-object tracking, speed estimation, red-light violation detection, and congestion analysis — built on YOLOv8 + supervision and tested on 3 real traffic video clips.

## System Features

| Feature | Implementation |
|---------|---------------|
| Vehicle detection | YOLOv8n — cars, trucks, buses, motorbikes |
| Multi-object tracking | ByteTrack (persistent IDs across frames) |
| Speed estimation | Pixel displacement × calibration factor → km/h |
| Red-light violation | Line zone crossing check during red phase |
| Congestion detection | Vehicle density per defined zone |
| Heatmap | Accumulated presence over full video |

## Architecture

```python
model   = YOLO('yolov8n.pt')
tracker = sv.ByteTrack()

# Define virtual counting lines / stop lines
stop_line   = sv.LineZone(start=Point(x1,y1), end=Point(x2,y2))
zone        = sv.PolygonZone(polygon=np.array([[...zone coords...]]))

def process_frame(frame, frame_idx):
    results  = model(frame, verbose=False)[0]
    detects  = sv.Detections.from_ultralytics(results)
    detects  = tracker.update_with_detections(detects)

    # Speed: compare centroid position to previous frame
    speeds   = estimate_speeds(detects, prev_detects, fps=30, calibration=0.05)

    # Red light: check if vehicle crosses stop_line while signal == RED
    crossed  = stop_line.trigger(detects)
    if red_phase and crossed.any():
        log_violation(detects[crossed])

    return annotate(frame, detects, speeds)
```

## Video Test Data

| File | Scene | Challenge |
|------|-------|-----------|
| `Traffic1.mp4` | Highway | High speed, lane changes |
| `Traffic2.mp4` | Intersection | Multiple directions, occlusion |
| `Traffic3.mp4` | Urban road | Mixed vehicle types, pedestrians |

## Speed Estimation
```
pixel_distance = ||centroid_t - centroid_(t-1)||  (pixels)
real_distance  = pixel_distance × (real_lane_width_m / lane_width_pixels)
speed_mps      = real_distance × fps
speed_kmh      = speed_mps × 3.6
```

Calibration requires knowing one real-world measurement in the scene (lane width, road marking distance).

## Key Challenge: Occlusion
When vehicles overlap, the tracker can lose a track or merge two tracks. ByteTrack handles this by:
1. **Kalman filter** — predicts where each object should be in the next frame
2. **Re-association** — if a track is lost for ≤N frames, it is held and matched when the vehicle reappears
