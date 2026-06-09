# OCR — Xerox Case Study

End-to-end document digitisation pipeline for scanned Xerox documents.

| File | Description |
|------|-------------|
| `OCR_Case_Xerox.ipynb` | Full OCR pipeline with EasyOCR |
| `document.png` | Original scan |
| `document_lossless.png` | Lossless compressed version |
| `document_lossy.png` | Lossy compressed version |
| `Case Xerox.pdf` | Case study brief |

**Pipeline:** Image quality assessment → compression artefact analysis → EasyOCR text extraction → post-processing cleanup

**Key learning:** Effect of compression on OCR accuracy; when lossy compression degrades text recognition.
