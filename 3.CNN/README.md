# CNN — Building Custom Convolution Filters

Before using `nn.Conv2d()`, understand what convolution actually does by building kernels manually in NumPy.

## What Is Convolution?
A convolution filter is a small matrix (kernel) that slides over an image, computing a weighted sum of the neighbourhood at each position. Different kernels extract different features.

## Kernels Implemented

### Edge Detection — Sobel
```python
sobel_x = [[-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]]   # detects vertical edges

sobel_y = [[-1, -2, -1],
            [0,  0,  0],
            [1,  2,  1]]  # detects horizontal edges
```

### Sharpening
```python
sharpen = [[ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]]   # centre weighted → enhances high-frequency detail
```

### Gaussian Blur
```python
gaussian = [[1, 2, 1],
             [2, 4, 2],   # / 16
             [1, 2, 1]]   # smooths noise by averaging with distance weighting
```

## Manual Convolution Implementation
```python
def convolve2d(image, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)))
    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    return output
```

## Key Insight
Every `nn.Conv2d` layer in a trained network has learned its own kernel weights — the training process discovers which patterns matter. By building filters manually, you see that a network is essentially learning which combinations of these low-level operations detect the features it needs.
