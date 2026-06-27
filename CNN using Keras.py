import numpy as np

# -------------------------------
# Input Image (4x4)
# -------------------------------
image = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9,10,11,12],
    [13,14,15,16]
])

# -------------------------------
# 2x2 Filter (Kernel)
# -------------------------------
kernel = np.array([
    [1, 0],
    [0, 1]
])

# -------------------------------
# Convolution Function
# -------------------------------
def convolution(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape

    output = np.zeros((h-kh+1, w-kw+1), dtype=int)

    for i in range(h-kh+1):
        for j in range(w-kw+1):
            region = image[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)

    return output

# -------------------------------
# Max Pooling Function
# -------------------------------
def max_pooling(feature_map, size=2, stride=2):
    h, w = feature_map.shape

    out_h = (h - size) // stride + 1
    out_w = (w - size) // stride + 1

    pooled = np.zeros((out_h, out_w), dtype=int)

    for i in range(out_h):
        for j in range(out_w):
            region = feature_map[
                i*stride:i*stride+size,
                j*stride:j*stride+size
            ]
            pooled[i, j] = np.max(region)

    return pooled

# -------------------------------
# Perform Operations
# -------------------------------
conv_output = convolution(image, kernel)
pool_output = max_pooling(conv_output)
flatten_output = pool_output.flatten()

# -------------------------------
# Display Results
# -------------------------------
print("Input Image:")
print(image)

print("\nKernel:")
print(kernel)

print("\nConvolution Output:")
print(conv_output)

print("\nMax Pooling Output:")
print(pool_output)

print("\nFlatten Output:")
print(flatten_output)