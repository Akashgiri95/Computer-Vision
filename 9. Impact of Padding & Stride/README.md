# Padding, Strides & Region-based Object Detection

| Notebook | Description |
|----------|-------------|
| `Padding & Strides in Convolution layer/Padding&Strides_CNN.ipynb` | Experiment: `valid` vs `same` padding, stride 1 vs 2 — visualise spatial dimension changes |
| `Region based Object Detection(Fast R-CNN)/RCNN_region_based_OD.ipynb` | Simplified Fast R-CNN: selective search → RoI pooling → classification head |

**Key formulae:**

```
Output size = floor((Input + 2×Padding - Kernel) / Stride) + 1
```

**Key learning:** Padding choice directly affects feature map size; strides trade resolution for speed. Fast R-CNN decouples region proposal from classification — enabling shared feature extraction.
