# Feature Descriptors — SIFT, SURF, ORB

| Notebook | Description |
|----------|-------------|
| `Feature_Descriptors.ipynb` | Extract keypoints with SIFT, SURF, ORB; match descriptors between two images using BFMatcher |

**Key learning:**
- **SIFT** — scale & rotation invariant, 128-dim descriptor, slower
- **SURF** — faster SIFT approximation using box filters
- **ORB** — binary descriptor (FAST + BRIEF), real-time capable, patent-free
- Lowe's ratio test for robust matching
