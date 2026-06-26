import numpy as np
from PIL import Image

I = np.array(Image.open('EE200_course_project_data_2026/Q1_data/ghost_signal_input.png'), dtype=np.float64)
F = np.fft.fft2(I)
F_s = np.fft.fftshift(F)
mag = np.abs(F_s)
mag_db = 20 * np.log10(mag + 1)

print("max:", np.max(mag_db))
print("min:", np.min(mag_db))
print("mean:", np.mean(mag_db))
print("median:", np.median(mag_db))
print("std:", np.std(mag_db))

# Test different thresholds
for p in [0.9, 0.8, 0.7]:
    t = p * np.max(mag_db)
    print(f"p={p} max -> {np.sum(mag_db > t)} spikes")

for sigma_mul in [3, 4, 5, 6, 10]:
    t = np.mean(mag_db) + sigma_mul * np.std(mag_db)
    print(f"mean + {sigma_mul}*std -> {np.sum(mag_db > t)} spikes")
