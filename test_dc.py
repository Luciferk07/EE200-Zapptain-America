import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

I = np.array(Image.open('EE200_course_project_data_2026/Q1_data/ghost_signal_input.png'), dtype=np.float64)
M, N = I.shape

F = np.fft.fft2(I)
F_s = np.fft.fftshift(F)
mag = np.abs(F_s)
mag_db = 20 * np.log10(mag + 1)

thresh = 50 * np.median(mag_db)
spike_coords = np.argwhere(mag_db > thresh)

# Filter out the DC component (center)
center_r, center_c = M // 2, N // 2
filtered_peaks = []
for r, c in spike_coords:
    if np.sqrt((r - center_r)**2 + (c - center_c)**2) > 20: # keep away from center
        filtered_peaks.append((r, c))

def notch_gauss(shape, peaks, sig):
    h = np.ones(shape)
    for (pr, pc) in peaks:
        yr, xr = np.ogrid[:shape[0], :shape[1]]
        h *= 1 - np.exp(-((yr-pr)**2 + (xr-pc)**2) / (2*sig**2))
    return h

mg = notch_gauss(I.shape, filtered_peaks, sig=3)
Ff = F_s * mg
rec_g = np.abs(np.fft.ifft2(np.fft.ifftshift(Ff)))

plt.imsave('test_recovery_no_dc.png', rec_g, cmap='gray')
print(f"Original peaks: {len(spike_coords)}, Filtered peaks: {len(filtered_peaks)}")
