# Advanced Labs

| Notebook / File | Description |
|-----------------|-------------|
| `Facial_Recognition_System.ipynb` | Production-style facial recognition: MTCNN detection → FaceNet embedding → cosine similarity verification |
| `Image Stiching/Deep Learning Stitching/ImageStitching_DeepLearning.ipynb` | CNN-based homography estimation for panoramic stitching |
| `Image Stiching/Manual_Implementation_and_OpenCV/Manual Image_Stitching_OpenCV.ipynb` | Classical: SIFT keypoints → RANSAC homography → perspective warp |
| `LAB_5.ipynb` | Mixed practicals |
| `TESLA PIPELINE.docx` | System design doc for autonomous vision pipeline |

## Image Stitching: DL vs Classical

| Approach | Method | Pros |
|----------|--------|------|
| Classical | SIFT + RANSAC + warp | No training data needed, interpretable |
| Deep Learning | CNN homography estimation | Handles textureless regions better |
