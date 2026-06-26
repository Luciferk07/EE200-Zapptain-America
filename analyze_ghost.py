"""Exploratory analysis of ghost_signal_input.png to find interference frequencies."""
import numpy as np
from PIL import Image
import json

# Load the image
img = Image.open(r'c:\Users\karti\OneDrive\Desktop\EE200 Project\EE200_course_project_data_2026\Q1_data\ghost_signal_input.png')
image = np.array(img, dtype=np.float64)
print(f"Image shape: {image.shape}, dtype: {image.dtype}")
print(f"Min: {image.min()}, Max: {image.max()}, Mean: {image.mean():.2f}")

M, N = image.shape

# Compute 2D FFT
F = np.fft.fft2(image)
F_shifted = np.fft.fftshift(F)
magnitude = np.abs(F_shifted)
mag_log = 20 * np.log10(magnitude + 1)

print(f"\nSpectrum shape: {magnitude.shape}")
print(f"DC component magnitude: {magnitude[M//2, N//2]:.2f}")

# Find the top peaks (excluding DC)
# Flatten and find top values
center_r, center_c = M // 2, N // 2

# Create a copy and zero out the DC region
mag_search = magnitude.copy()
# Zero out a small region around DC to ignore it
dc_radius = 5
for r in range(max(0, center_r-dc_radius), min(M, center_r+dc_radius+1)):
    for c in range(max(0, center_c-dc_radius), min(N, center_c+dc_radius+1)):
        mag_search[r, c] = 0

# Find top 20 peaks
peaks = []
mag_temp = mag_search.copy()
for i in range(20):
    idx = np.unravel_index(np.argmax(mag_temp), mag_temp.shape)
    val = mag_temp[idx]
    # Relative to center
    rel_r = idx[0] - center_r
    rel_c = idx[1] - center_c
    peaks.append({
        'row': int(idx[0]), 'col': int(idx[1]),
        'rel_row': int(rel_r), 'rel_col': int(rel_c),
        'magnitude': float(val),
        'mag_log': float(20 * np.log10(val + 1))
    })
    # Zero out a small region around this peak
    r1, r2 = max(0, idx[0]-3), min(M, idx[0]+4)
    c1, c2 = max(0, idx[1]-3), min(N, idx[1]+4)
    mag_temp[r1:r2, c1:c2] = 0

print("\n=== TOP 20 INTERFERENCE PEAKS (excluding DC) ===")
print(f"{'#':>3} {'Row':>5} {'Col':>5} {'RelR':>6} {'RelC':>6} {'Magnitude':>12} {'dB':>8}")
for i, p in enumerate(peaks):
    print(f"{i+1:3d} {p['row']:5d} {p['col']:5d} {p['rel_row']:6d} {p['rel_col']:6d} {p['magnitude']:12.2f} {p['mag_log']:8.2f}")

# Also check: what's the ratio of peak magnitudes to average magnitude?
avg_mag = np.mean(magnitude)
print(f"\nAverage magnitude: {avg_mag:.2f}")
print(f"DC magnitude: {magnitude[center_r, center_c]:.2f}")
for i, p in enumerate(peaks[:6]):
    print(f"Peak {i+1} is {p['magnitude']/avg_mag:.1f}x average magnitude")
