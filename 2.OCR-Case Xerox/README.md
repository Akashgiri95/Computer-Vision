# OCR — Xerox JBIG2 Case Study

Investigation of the famous **Xerox WorkCentre bug** where scanned documents had digits silently substituted during compression. This is a real-world case where a compression algorithm broke document integrity in a way humans couldn't detect but machines couldn't survive.

## The Bug

Xerox's JBIG2 compression identifies visually similar character patterns and substitutes them with a single stored template. In some documents, the digit **6** was being replaced by **8** (and similar swaps) during scan → print cycles — changing a patient's medication dose from 6mg to 8mg without any visible artefact.

**Why humans missed it:** The substituted glyph looks correct to a human reader.
**Why OCR fails:** The underlying pixel pattern is wrong — the bit-level representation was corrupted.

## What This Notebook Explores

```
Original document scan
    ↓
Lossless vs Lossy compression comparison
    ↓
Connected Component Analysis (extract individual character blobs)
    ↓
Shape Descriptors: Hu Moments + aspect ratio + size
    ↓
Distance-based character matching (detect substitutions)
    ↓
EasyOCR on all three versions → accuracy comparison
```

## Key Implementation

### Shape Descriptors (Hu Moments)
```python
def shape_descriptor(img):
    moments  = cv2.moments(img)
    hu       = cv2.HuMoments(moments).flatten()
    hu       = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)  # log-scale
    h, w     = img.shape
    return np.concatenate([hu, [w, h, w/h]])   # 7 Hu + size features
```

Hu Moments are **rotation, scale, and translation invariant** — ideal for detecting when two glyphs are the "same shape" even after compression distortion.

### Distance Metric
```python
def distance(d1, d2):
    return np.linalg.norm(d1 - d2)   # Euclidean distance in feature space
```

Pairs of characters with very low distance but different EasyOCR readings = suspected JBIG2 substitutions.

## Files

| File | Description |
|------|-------------|
| `OCR_Case_Xerox.ipynb` | Full investigation pipeline |
| `document.png` | Original scan |
| `document_lossless.png` | Lossless compressed |
| `document_lossy.png` | Lossy compressed (shows substitutions) |
| `Case Xerox.pdf` | Case study brief |

## Key Takeaway

Compression artefacts are not always visible. For documents where accuracy is critical (medical records, financial statements, legal documents) — verify OCR output from compressed sources. The JBIG2 bug affected Xerox scanners globally from 2010 onwards.
