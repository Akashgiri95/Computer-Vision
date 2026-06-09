# Traffic Monitoring & Video Analytics

Real-time vehicle detection, counting, and tracking using YOLOv8 on actual traffic footage.

| File | Description |
|------|-------------|
| `Traffic Monitoring.ipynb` | Vehicle detection + counting on 3 traffic clips |
| `Video_analytics.ipynb` | Extended: speed estimation, zone analytics, heatmaps |
| `Traffic1.mp4` | Highway footage (sample 1) |
| `Traffic2.mp4` | Intersection footage |
| `Traffic3.mp4` | Urban road footage |
| `yolov8n.pt` | YOLOv8 nano weights |

**Pipeline:** Video frame extraction → YOLOv8 detection → ByteTrack tracking → counting line logic → analytics dashboard

**Key learning:** Real-world system design for traffic AI — handling occlusion, direction classification, false positive filtering.
