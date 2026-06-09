# Feature Descriptors — SIFT, SURF, ORB

Detect, describe, and match keypoints between two images using three classical descriptors. Feature matching is the foundation of image stitching, 3D reconstruction, and object recognition.

## What Is a Feature Descriptor?
A feature descriptor converts a keypoint location into a fixed-length vector that describes the local appearance. A good descriptor is invariant to scale, rotation, and lighting changes.

## The Three Descriptors

### SIFT (Scale-Invariant Feature Transform)
1. Build a **Gaussian scale-space pyramid** (multiple blurs)
2. Find local extrema in the **Difference of Gaussians** (DoG) — these are the keypoints
3. Assign orientation from gradient histogram (rotation invariant)
4. Build a **128-dimensional descriptor** from 4×4 grid of 8-bin gradient histograms

```python
sift = cv2.SIFT_create()
kp, desc = sift.detectAndCompute(gray_img, None)
# kp: list of KeyPoint objects (x, y, scale, angle)
# desc: (N×128) float32 array
```

### SURF (Speeded-Up Robust Features)
Approximates SIFT using **box filters** (integral images) for faster computation. Uses 64-dim descriptors (or 128 for extended). ~3× faster than SIFT.

### ORB (Oriented FAST + Rotated BRIEF)
Combines **FAST** keypoint detection with **BRIEF** binary descriptors. No patents, extremely fast, binary descriptor enables Hamming distance matching.

```python
orb = cv2.ORB_create(nfeatures=500)
kp, desc = orb.detectAndCompute(gray_img, None)
# desc: (N×32) uint8 — each row is 256 bits packed into 32 bytes
```

## Matching Pipeline

```python
# BFMatcher with Lowe's ratio test (filters ambiguous matches)
bf      = cv2.BFMatcher(cv2.NORM_L2)          # SIFT/SURF
# bf    = cv2.BFMatcher(cv2.NORM_HAMMING)     # ORB
matches = bf.knnMatch(desc1, desc2, k=2)

# Lowe's ratio test — keep match only if best match is clearly better than second best
good = [m for m, n in matches if m.distance < 0.75 * n.distance]
```

## Descriptor Comparison

| | SIFT | SURF | ORB |
|-|------|------|-----|
| Invariance | Scale + Rotation | Scale + Rotation | Rotation |
| Descriptor | 128-dim float | 64-dim float | 256-bit binary |
| Matching | L2 distance | L2 distance | Hamming distance |
| Speed | Slow | Medium | Very fast |
| Patent | Free (expired 2020) | Yes | No |

**Conclusion:** Use ORB for real-time applications; SIFT when accuracy matters and speed is not critical.
