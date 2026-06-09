# Advanced Labs

Production-level implementations: complete facial recognition pipeline, image stitching in two paradigms, and a system design document.

---

## Lab: Complete Facial Recognition System

A step up from the triplet-loss notebook — a full working system with all components integrated.

### Pipeline
```
Input image / webcam frame
    ↓
Face Detection (MTCNN)
    → Detect face bounding boxes
    → Facial landmark prediction (eyes, nose, mouth corners)
    ↓
Alignment
    → Rotate and crop face to canonical orientation using landmarks
    → Eliminates pose variation before embedding
    ↓
Embedding (FaceNet)
    → 128-dimensional L2-normalised vector
    ↓
Verification / Identification
    → Cosine similarity against enrolled face database
    → Decision at tuned similarity threshold
```

**MTCNN** (Multi-task Cascaded Convolutional Network) runs three stages: P-Net (proposal), R-Net (refine), O-Net (output with landmarks). Accurate for varied poses, lighting, and occlusion.

---

## Lab: Image Stitching — Classical vs Deep Learning

### Classical Method (SIFT + RANSAC + Homography)
```python
# 1. Detect and match keypoints
sift   = cv2.SIFT_create()
kp1, d1 = sift.detectAndCompute(img1_gray, None)
kp2, d2 = sift.detectAndCompute(img2_gray, None)
matches = BFMatcher().knnMatch(d1, d2, k=2)
good    = [m for m, n in matches if m.distance < 0.75*n.distance]

# 2. Estimate homography (RANSAC filters outliers)
src_pts  = np.float32([kp1[m.queryIdx].pt for m in good])
dst_pts  = np.float32([kp2[m.trainIdx].pt for m in good])
H, mask  = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 3. Warp and blend
warped   = cv2.warpPerspective(img1, H, (w1+w2, h1))
result   = blend(warped, img2)
```

**RANSAC** (Random Sample Consensus) randomly samples 4 point correspondences, computes the homography, and counts inliers. The best homography (most inliers) is returned — making it robust to false keypoint matches.

### Deep Learning Method (CNN Homography Estimation)
Instead of explicit keypoints, a CNN directly regresses the 4-point homography parameterisation from image pair patches.

| | Classical | Deep Learning |
|-|-----------|---------------|
| Requires textures | Yes (needs keypoints) | No |
| Accuracy on low-texture scenes | Low | Higher |
| Interpretability | High | Low |
| Training data needed | None | Yes |

---

## TESLA Vision Pipeline (System Design)
Design document covering the full autonomous vehicle perception stack:
- Multi-camera input fusion
- Real-time object detection + depth estimation
- Lane detection and tracking
- Sensor fusion (camera + radar)
- Occupancy grid generation
