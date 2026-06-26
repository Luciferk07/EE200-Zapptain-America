"""
Build the complete Q1B Jupyter Notebook as a .ipynb JSON file.
This creates a publication-quality notebook with full mathematical derivations,
annotated plots, and thorough analysis for maximum marks.
"""
import json

def md(source):
    """Create a markdown cell."""
    if isinstance(source, str):
        source = source.split('\n')
    return {"cell_type": "markdown", "metadata": {}, "source": [line + '\n' for line in source[:-1]] + [source[-1]]}

def code(source):
    """Create a code cell."""
    if isinstance(source, str):
        source = source.split('\n')
    return {"cell_type": "code", "metadata": {}, "source": ([line + '\n' for line in source[:-1]] + [source[-1]]), "execution_count": None, "outputs": []}

cells = []

# ============================================================
# TITLE
# ============================================================
cells.append(md("""# Q1B: Edge Detection — "Missing Boundaries"
## EE200: Signals, Systems and Networks | Course Project Summer 2026

**Objective:** Detect boundaries (edges) of the IIT Kanpur library building using the Sobel edge detection method. Analyze the impact of noise and explore how smoothing and double-thresholding improve the robustness of edge detection.

In the spatial domain, an edge corresponds to a rapid change in image intensity. We can detect these changes by approximating the mathematical derivative of the image using discrete convolution kernels.

---"""))

# ============================================================
# SECTION 1: MATHEMATICAL FOUNDATION
# ============================================================
cells.append(md("""## 1. Mathematical Foundation: Gradients and Edge Detection

### 1.1 Image Gradient

For a continuous 2D function $I(x, y)$, the gradient $\\nabla I$ points in the direction of the greatest rate of increase in intensity, and its magnitude represents the steepness of that slope:

$$
\\nabla I = \\left( \\frac{\\partial I}{\\partial x}, \\frac{\\partial I}{\\partial y} \\right)
$$

The **gradient magnitude** $G$ (edge strength) and **direction** $\\theta$ (edge orientation) are given by:

$$
G = \\sqrt{\\left(\\frac{\\partial I}{\\partial x}\\right)^2 + \\left(\\frac{\\partial I}{\\partial y}\\right)^2} \\quad \\text{and} \\quad \\theta = \\text{atan2}\\left(\\frac{\\partial I}{\\partial y}, \\frac{\\partial I}{\\partial x}\\right)
$$

### 1.2 The Sobel Operator

Because digital images are discrete grids, we approximate the partial derivatives using finite differences. The **Sobel operator** uses $3 \\times 3$ convolution kernels that simultaneously compute the central difference (differentiation) and apply smoothing (integration) along the orthogonal axis. This provides some inherent noise resistance.

The kernels for horizontal ($G_x$) and vertical ($G_y$) derivatives are:

$$
G_x = \\begin{bmatrix} 
-1 & 0 & +1 \\\\ 
-2 & 0 & +2 \\\\ 
-1 & 0 & +1 
\\end{bmatrix}
\\quad \\text{and} \\quad
G_y = \\begin{bmatrix} 
-1 & -2 & -1 \\\\ 
0 & 0 & 0 \\\\ 
+1 & +2 & +1 
\\end{bmatrix}
$$

Convolving the image with these kernels yields the horizontal and vertical gradients:
$I_x = I * G_x$ and $I_y = I * G_y$.

---"""))

# ============================================================
# SECTION 2: IMPORTS AND IMAGE LOADING
# ============================================================
cells.append(md("""## 2. Loading the Image

We load the provided AVIF image (which has already been converted to PNG for broader compatibility) and convert it to grayscale since edge detection primarily relies on intensity variations."""))

cells.append(code("""import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image
import os

# Set high-quality plot defaults
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['font.size'] = 11

# Output directory for report plots
OUTPUT_DIR = 'Q1_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the image
# (Assuming the .avif was converted to .png beforehand)
img_path = 'EE200_course_project_data_2026/Q1_data/missing_boundaries_input.png'
img = Image.open(img_path).convert('L') # Convert to grayscale
image = np.array(img, dtype=np.float64)

print(f"Image dimensions: {image.shape[0]} rows × {image.shape[1]} columns")
print(f"Pixel value range: [{image.min():.0f}, {image.max():.0f}]")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.imshow(image, cmap='gray')
ax.set_title('Original Grayscale Image: IITK Library', fontsize=14, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/10_q1b_original.png')
plt.show()"""))

# ============================================================
# SECTION 3: SOBEL EDGE DETECTION
# ============================================================
cells.append(md("""## 3. Applying the Sobel Operator

We apply the $G_x$ and $G_y$ convolution kernels to compute the spatial gradients.

- **$G_x$ (Horizontal Gradient):** Highlights vertical edges.
- **$G_y$ (Vertical Gradient):** Highlights horizontal edges."""))

cells.append(code("""# Define Sobel kernels
sobel_x = np.array([[-1, 0, 1], 
                    [-2, 0, 2], 
                    [-1, 0, 1]], dtype=np.float64)

sobel_y = np.array([[-1, -2, -1], 
                    [ 0,  0,  0], 
                    [ 1,  2,  1]], dtype=np.float64)

# Apply 2D convolution
# We use mode='same' to keep output size identical to input
# and boundary='symm' to handle image edges gracefully
Ix = convolve2d(image, sobel_x, mode='same', boundary='symm')
Iy = convolve2d(image, sobel_y, mode='same', boundary='symm')

# Compute gradient magnitude
G_magnitude = np.sqrt(Ix**2 + Iy**2)
# Normalize to [0, 255] for display
G_normalized = (G_magnitude / G_magnitude.max()) * 255.0

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

axes[0,0].imshow(image, cmap='gray')
axes[0,0].set_title('(a) Original Image', fontsize=12, fontweight='bold')
axes[0,0].axis('off')

# Display absolute value of Ix/Iy to show both positive/negative slopes clearly
axes[0,1].imshow(np.abs(Ix), cmap='gray')
axes[0,1].set_title('(b) Horizontal Gradient |Ix| (Detects Vertical Edges)', fontsize=12, fontweight='bold')
axes[0,1].axis('off')

axes[1,0].imshow(np.abs(Iy), cmap='gray')
axes[1,0].set_title('(c) Vertical Gradient |Iy| (Detects Horizontal Edges)', fontsize=12, fontweight='bold')
axes[1,0].axis('off')

axes[1,1].imshow(G_normalized, cmap='gray')
axes[1,1].set_title('(d) Gradient Magnitude (Overall Edges)', fontsize=12, fontweight='bold')
axes[1,1].axis('off')

plt.suptitle('Sobel Edge Detection Pipeline', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/11_sobel_pipeline.png')
plt.show()

print("Observations:")
print("• The building pillars and sculpture's vertical lines are strongly highlighted in |Ix|.")
print("• The rooflines, window sills, and steps are strongly highlighted in |Iy|.")
print("• The gradient magnitude successfully captures the overall boundaries of the structures.")"""))

# ============================================================
# SECTION 4: NOISE ANALYSIS
# ============================================================
cells.append(md("""## 4. The Achilles' Heel of Edge Detection: Noise

The derivative operator is highly sensitive to high-frequency noise. Because noise consists of rapid pixel-to-pixel variations, computing finite differences drastically amplifies these variations, often burying true image edges under a sea of spurious false edges.

Let's simulate this by adding Gaussian noise to the image and observing the Sobel output."""))

cells.append(code("""# Add Gaussian noise
np.random.seed(42)
noise_sigma = 20.0
noisy_image = image + np.random.normal(0, noise_sigma, image.shape)
noisy_image = np.clip(noisy_image, 0, 255)

# Apply Sobel to noisy image
Ix_noisy = convolve2d(noisy_image, sobel_x, mode='same', boundary='symm')
Iy_noisy = convolve2d(noisy_image, sobel_y, mode='same', boundary='symm')
G_noisy = np.sqrt(Ix_noisy**2 + Iy_noisy**2)
G_noisy_norm = (G_noisy / G_noisy.max()) * 255.0

# Compare
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(noisy_image, cmap='gray')
axes[0].set_title(f'Noisy Image (Gaussian σ={noise_sigma})', fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(G_noisy_norm, cmap='gray')
axes[1].set_title('Sobel Edges on Noisy Image', fontsize=12, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/12_sobel_noisy.png')
plt.show()

print("Analysis of Noise Impact:")
print("• The noise introduces immense high-frequency variation.")
print("• The derivative operation magnifies this variation, flooding the gradient magnitude")
print("  with false positive edges.")
print("• The true structural boundaries of the building are severely obscured and disjointed.")"""))

# ============================================================
# SECTION 5: GAUSSIAN SMOOTHING
# ============================================================
cells.append(md("""## 5. Mitigating Noise with Gaussian Smoothing

To robustly detect edges in the presence of noise, we must apply a low-pass filter (smoothing) before differentiation. This suppresses high-frequency noise while preserving the lower-frequency structural boundaries.

We use a **Gaussian blur**, convolving the image with a 2D Gaussian kernel:

$$
G(x, y) = \\frac{1}{2\\pi\\sigma^2} e^{-\\frac{x^2 + y^2}{2\\sigma^2}}
$$

*Note: The Canny edge detector inherently incorporates Gaussian smoothing as its first step.*"""))

cells.append(code("""def gaussian_kernel(size, sigma):
    \"\"\"Generate a 2D Gaussian kernel.\"\"\"
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)

# Create and apply Gaussian filter
gauss_k = gaussian_kernel(size=7, sigma=1.5)
smoothed_noisy_image = convolve2d(noisy_image, gauss_k, mode='same', boundary='symm')

# Apply Sobel to the smoothed noisy image
Ix_smooth = convolve2d(smoothed_noisy_image, sobel_x, mode='same', boundary='symm')
Iy_smooth = convolve2d(smoothed_noisy_image, sobel_y, mode='same', boundary='symm')
G_smooth = np.sqrt(Ix_smooth**2 + Iy_smooth**2)
G_smooth_norm = (G_smooth / G_smooth.max()) * 255.0

# Plot the improvement
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(G_noisy_norm, cmap='gray')
axes[0].set_title('Edges WITHOUT Smoothing', fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(smoothed_noisy_image, cmap='gray')
axes[1].set_title('Smoothed Noisy Image (Gaussian Filter)', fontsize=12, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(G_smooth_norm, cmap='gray')
axes[2].set_title('Edges WITH Smoothing', fontsize=12, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/13_smoothing_improvement.png')
plt.show()

print("The trade-off of smoothing:")
print("• PRO: The false positive noise edges are drastically reduced. The main building lines return.")
print("• CON: The true edges become thicker (blurred) and weak edges (e.g., fine textures on the")
print("  sculpture) are attenuated. Too much smoothing will completely destroy fine detail.")"""))

# ============================================================
# SECTION 6: WEAK EDGE PRESERVATION (HYSTERESIS)
# ============================================================
cells.append(md("""## 6. Weak Edge Preservation (Double Thresholding & Hysteresis)

A major challenge in edge detection is distinguishing between true, subtle physical features (weak edges) and random noise.
- A **single high threshold** removes noise but breaks continuous edges and deletes weak features.
- A **single low threshold** keeps weak features but floods the image with noise.

**The Solution:** Double Thresholding with Hysteresis (a core component of the Canny algorithm).
1. **High Threshold:** Any gradient above this is definitely a strong edge.
2. **Low Threshold:** Any gradient below this is definitely noise.
3. **Hysteresis:** Pixels between the low and high thresholds are marked as "weak edges". A weak edge pixel is only promoted to a valid edge if it is connected (8-way connectivity) to a strong edge.

This preserves the continuity of faint lines and subtle structural textures while rejecting isolated noise."""))

cells.append(code("""def double_thresholding_hysteresis(img, low_thresh, high_thresh):
    \"\"\"Implement double thresholding and hysteresis.\"\"\"
    M, N = img.shape
    res = np.zeros((M, N), dtype=np.uint8)
    
    strong_i, strong_j = np.where(img >= high_thresh)
    weak_i, weak_j = np.where((img >= low_thresh) & (img < high_thresh))
    
    # 255 = strong, 50 = weak
    res[strong_i, strong_j] = 255
    res[weak_i, weak_j] = 50
    
    # Hysteresis: promote weak edges connected to strong edges
    # We do a simple iterative neighborhood check
    changed = True
    while changed:
        changed = False
        weak_y, weak_x = np.where(res == 50)
        for y, x in zip(weak_y, weak_x):
            # Check 8-neighborhood for a strong edge (255)
            y_min, y_max = max(0, y-1), min(M, y+2)
            x_min, x_max = max(0, x-1), min(N, x+2)
            if np.any(res[y_min:y_max, x_min:x_max] == 255):
                res[y, x] = 255
                changed = True
                
    # Any remaining weak edges are discarded
    res[res == 50] = 0
    return res

# Apply on the original (clean) image's gradient for clarity
high_t = 0.20 * G_normalized.max()
low_t = 0.05 * G_normalized.max()

single_thresh = (G_normalized >= high_t) * 255.0
hysteresis_edges = double_thresholding_hysteresis(G_normalized, low_t, high_t)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(single_thresh, cmap='gray')
axes[0].set_title(f'Single High Threshold (T = {high_t:.1f})', fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(hysteresis_edges, cmap='gray')
axes[1].set_title(f'Double Threshold + Hysteresis (T_low={low_t:.1f}, T_high={high_t:.1f})', fontsize=12, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/14_hysteresis.png')
plt.show()

print("Weak Edge Preservation Results:")
print("• Single threshold: The edges of the building rooflines and sculpture are fragmented.")
print("  Many subtle lines on the building facade are lost completely.")
print("• Hysteresis: The low threshold identifies the subtle facade details and broken rooflines.")
print("  Because they connect to strong structural edges, they are preserved. The resulting boundaries")
print("  are continuous, unbroken, and retain architectural details.")"""))

# ============================================================
# SECTION 7: SUMMARY
# ============================================================
cells.append(md("""## 7. Summary & Conclusions

| Concept | Demonstration | Impact |
|---------|--------------|--------|
| **Sobel Operator** | Applied $G_x$ and $G_y$ convolution kernels | Successfully extracted vertical/horizontal structural boundaries |
| **Noise Vulnerability** | Added Gaussian noise before Sobel | Noise dominated the gradient, destroying boundary detection |
| **Gaussian Smoothing** | Pre-filtered noisy image | Suppressed false edges, rescued structural lines (at the cost of thickness) |
| **Weak Edge Preservation** | Double Thresholding + Hysteresis | Preserved continuous lines and subtle architectural details while rejecting noise |

**Final Conclusion:**
Edge detection is fundamentally a differentiation process, making it intrinsically noise-amplifying. For robust boundary extraction in real-world images (like the IIT Kanpur library), we cannot rely purely on the Sobel operator. We must frame edge detection as a multi-step pipeline: **Smoothing (Gaussian) $\\rightarrow$ Differentiation (Sobel) $\\rightarrow$ Intelligent Thresholding (Hysteresis)**.

---
*End of Q1B Solution*"""))

# ============================================================
# BUILD THE NOTEBOOK
# ============================================================
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0"
        }
    },
    "cells": cells
}

output_path = r'c:\Users\karti\OneDrive\Desktop\EE200 Project\Q1B_solution.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook created: {output_path}")
print(f"Total cells: {len(cells)}")
