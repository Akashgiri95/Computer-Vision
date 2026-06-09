# Corner Detection — Harris & FAST from Scratch

Both detectors implemented **entirely in NumPy** — no `cv2.cornerHarris()` or `cv2.FastFeatureDetector`. Building from the maths develops real understanding.

## What Is a Corner?
A corner is a point where the image intensity changes significantly in **two directions**. Edges change in one direction. Flat regions change in neither.

This is captured by the **structure tensor** (second-moment matrix):

```
M = Σ_window [ [Ix²,  IxIy],
               [IxIy, Iy² ] ]
```
where Ix, Iy are image gradients.

## Harris Corner Detector

```python
def harris_corners_from_scratch(gray_img, block_size=3, ksize=3, k=0.04, threshold_ratio=0.01):
    # Step 1: Compute gradients with Sobel
    Ix = sobel(gray, axis=1)
    Iy = sobel(gray, axis=0)

    # Step 2: Compute products, average in local window
    Ix2  = gaussian_filter(Ix * Ix,  sigma=block_size)
    Iy2  = gaussian_filter(Iy * Iy,  sigma=block_size)
    IxIy = gaussian_filter(Ix * Iy,  sigma=block_size)

    # Step 3: Corner response function
    det   = Ix2 * Iy2 - IxIy ** 2
    trace = Ix2 + Iy2
    R     = det - k * trace ** 2   # k=0.04 is Harris's recommended constant

    # Step 4: Threshold and NMS
    threshold = threshold_ratio * R.max()
    corners   = np.argwhere(R > threshold)
    return corners
```

**Interpretation of R:**
- `R >> 0` → Corner (large eigenvalues in both directions)
- `R < 0` → Edge (one large eigenvalue)
- `|R| ≈ 0` → Flat region

## FAST Corner Detector

Much simpler: examine 16 pixels on a circle of radius 3 around the candidate point. If N or more **consecutive** pixels are all brighter than `I_centre + threshold` (or all darker), it's a corner. Default N=12.

```python
def fast_corners_from_scratch(gray_img, threshold=25, n=12):
    # For each non-border pixel:
    #   Check 16 pixels on the Bresenham circle
    #   Count consecutive pixels above/below threshold
    #   If longest run >= n: mark as corner
```

**Why FAST is fast:** It uses an early exit — check pixels 1, 9, 5, 13 first. If fewer than 3 of these 4 pass the threshold, it can't be a corner and the full check is skipped.

## Comparison

| | Harris | FAST |
|-|--------|------|
| Method | Eigenvalue analysis | Pixel ring test |
| Accuracy | Higher (subpixel) | Lower |
| Speed | Slow | Very fast (real-time) |
| Use case | Offline feature matching | ORB, real-time SLAM |
