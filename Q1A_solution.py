#!/usr/bin/env python
# coding: utf-8

# # Q1A: The Ghost Signal — Frequency Forensics
# Recover a hidden message from a corrupted image using 2D FFT and notch filtering.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['font.size'] = 11

OUT = 'Q1_outputs'
os.makedirs(OUT, exist_ok=True)

I = np.array(Image.open('EE200_course_project_data_2026/Q1_data/ghost_signal_input.png'), dtype=np.float64)
M, N = I.shape

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(I, cmap='gray')
ax.set_title('Corrupted Image')
plt.tight_layout()
plt.savefig(f'{OUT}/01_corrupted_image.png', bbox_inches='tight')
plt.show()


# ## 2D FFT and Spectrum Analysis

# In[2]:


F = np.fft.fft2(I)
F_s = np.fft.fftshift(F)
mag = np.abs(F_s)
mag_db = 20 * np.log10(mag + 1)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].imshow(mag_db, cmap='hot')
axes[0].set_title('Corrupted — Log Magnitude Spectrum')
axes[0].axis('off')
axes[1].imshow(np.zeros_like(mag_db), cmap='gray')
axes[1].set_title('Clean Spectrum (not available separately)')
axes[1].axis('off')
plt.tight_layout()
plt.savefig(f'{OUT}/02_spectrum_comparison.png', bbox_inches='tight')
plt.show()


# ## Detecting Interference Spikes

# In[3]:


thresh = 50 * np.median(mag_db)
spike_coords = np.argwhere(mag_db > thresh)

fig, ax = plt.subplots(figsize=(12, 7))
ax.imshow(mag_db, cmap='hot')
for (r, c) in spike_coords:
    ax.add_patch(plt.Circle((c, r), 8, color='cyan', fill=False, lw=1.5))
ax.set_title('Annotated Spectrum — Interference Spikes')
ax.axis('off')
plt.tight_layout()
plt.savefig(f'{OUT}/03_annotated_spectrum.png', bbox_inches='tight')
plt.show()

peaks = spike_coords
print(f'Detected {len(peaks)} spike pixels')


# ## Notch Filter Design
# Binary (hard cutoff) and Gaussian (smooth) notch filters.

# In[4]:


def notch_binary(shape, peaks, r):
    h = np.ones(shape)
    for (pr, pc) in peaks:
        yr, xr = np.ogrid[:shape[0], :shape[1]]
        h[np.sqrt((yr-pr)**2 + (xr-pc)**2) <= r] = 0
    return h

def notch_gauss(shape, peaks, sig):
    h = np.ones(shape)
    for (pr, pc) in peaks:
        yr, xr = np.ogrid[:shape[0], :shape[1]]
        h *= 1 - np.exp(-((yr-pr)**2 + (xr-pc)**2) / (2*sig**2))
    return h

def reconstruct(Fs, mask):
    Ff = Fs * mask
    return Ff, np.abs(np.fft.ifft2(np.fft.ifftshift(Ff)))

r, sig = 8, 3
mb = notch_binary(I.shape, peaks, r)
mg = notch_gauss(I.shape, peaks, sig)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(mb, cmap='gray'); axes[0].set_title('Binary Notch'); axes[0].axis('off')
axes[1].imshow(mg, cmap='gray'); axes[1].set_title('Gaussian Notch'); axes[1].axis('off')
plt.tight_layout()
plt.savefig(f'{OUT}/04_filter_masks.png', bbox_inches='tight')
plt.show()


# ## Recovery — Binary Filter

# In[5]:


Fb, rec_b = reconstruct(F_s, mb)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax_, arr, ttl in zip(axes,
    [I, mag_db, 20*np.log10(np.abs(Fb)+1), rec_b],
    ['Corrupted', 'Spectrum', 'Filtered Spectrum', 'Recovered (Binary)']):
    ax_.imshow(arr, cmap='gray' if ttl != 'Spectrum' and ttl != 'Filtered Spectrum' else 'hot')
    ax_.set_title(ttl); ax_.axis('off')
plt.tight_layout()
plt.savefig(f'{OUT}/05_recovery_binary_4panel.png', bbox_inches='tight')
plt.show()


# ## Recovery — Gaussian Filter

# In[6]:


Fg, rec_g = reconstruct(F_s, mg)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax_, arr, ttl in zip(axes,
    [I, mag_db, 20*np.log10(np.abs(Fg)+1), rec_g],
    ['Corrupted', 'Spectrum', 'Filtered Spectrum', 'Recovered (Gaussian)']):
    ax_.imshow(arr, cmap='gray' if 'Spectrum' not in ttl else 'hot')
    ax_.set_title(ttl); ax_.axis('off')
plt.tight_layout()
plt.savefig(f'{OUT}/06_recovery_gaussian_4panel.png', bbox_inches='tight')
plt.show()


# ## Binary vs Gaussian — Direct Comparison

# In[7]:


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].imshow(I, cmap='gray'); axes[0].set_title('Corrupted'); axes[0].axis('off')
axes[1].imshow(rec_b, cmap='gray'); axes[1].set_title(f'Binary (r={r})'); axes[1].axis('off')
axes[2].imshow(rec_g, cmap='gray'); axes[2].set_title(f'Gaussian (σ={sig})'); axes[2].axis('off')
plt.suptitle('Recovery Quality Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/07_binary_vs_gaussian.png', bbox_inches='tight')
plt.show()


# ## Effect of Notch Radius

# In[8]:


radii = [2, 5, 10, 20, 40]
fig, axes = plt.subplots(1, len(radii), figsize=(20, 4))
for ax_, rv in zip(axes, radii):
    _, rec = reconstruct(F_s, notch_binary(I.shape, peaks, rv))
    ax_.imshow(rec, cmap='gray'); ax_.set_title(f'r={rv}'); ax_.axis('off')
plt.suptitle('Effect of Notch Radius', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT}/08_radius_comparison.png', bbox_inches='tight')
plt.show()


# ## Final Recovered Message

# In[9]:


fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(rec_g, cmap='gray')
ax.set_title('RECOVERED: QUIZ 2 ON 7th JULY IN TUTORIAL HOURS',
             fontsize=14, fontweight='bold', color='darkgreen')
ax.axis('off')
plt.tight_layout()
plt.savefig(f'{OUT}/09_recovered_message.png', bbox_inches='tight')
plt.show()

