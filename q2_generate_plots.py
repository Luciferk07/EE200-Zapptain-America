import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

ecg = np.load('EE200_course_project_data_2026/Q2_data/patient_ecg.npy')
tmpl = np.load('EE200_course_project_data_2026/Q2_data/template.npy')
fs = 250
N = len(ecg)
L = len(tmpl)
t_ax = np.arange(N) / fs

def get_onset(ecg, tmpl, thr=0.5):
    L = len(tmpl)
    N = len(ecg)
    nt = np.linalg.norm(tmpl)
    onset = -1
    scores, idxs = [], []
    for m in range(0, N - L + 1, L):
        seg = ecg[m : m + L]
        ns = np.linalg.norm(seg)
        rho = float(np.dot(tmpl, seg) / (nt * ns)) if ns > 0 else 0.0
        scores.append(rho)
        idxs.append(m)
        if rho < thr and onset == -1:
            onset = m
    return onset, scores, idxs

idx, scores, idxs = get_onset(ecg, tmpl, 0.5)
t_onset = idx / fs if idx != -1 else None

fig, ax = plt.subplots(2, 1, figsize=(14, 7))
ax[0].plot(t_ax, ecg, color='#475569', lw=0.8)
if t_onset:
    ax[0].axvline(t_onset, color='red', lw=1.8, linestyle='--')
    ax[0].axvspan(t_onset, N/fs, alpha=0.07, color='red')
ax[0].set_xlim(0, N/fs)

t_beats = [m/fs for m in idxs]
cols = ['green' if r >= 0.5 else 'red' for r in scores]
ax[1].plot(t_beats, scores, color='#818cf8', lw=1.4, zorder=1)
ax[1].scatter(t_beats, scores, c=cols, s=70, zorder=2)
ax[1].axhline(0.5, color='orange', lw=1.5, linestyle='--')
ax[1].axhline(0.0, color='gray', lw=0.8, linestyle=':')
if t_onset:
    ax[1].axvline(t_onset, color='red', lw=1.8, linestyle='--')
ax[1].set_xlim(0, N/fs)
ax[1].set_ylim(-1.15, 1.15)
plt.tight_layout()
plt.savefig('q2_correlation_plot.png', dpi=150)

f, t, Sxx = spectrogram(ecg, fs=fs, nperseg=200, noverlap=150, window='hann')
fig2, ax3 = plt.subplots(figsize=(14, 5))
pcm = ax3.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='inferno', vmin=-80, vmax=-20)
ax3.set_ylim(0, 25)
for k in range(1, 20):
    fk = k * 1.25
    if fk > 25: break
    ax3.axhline(fk, color='cyan', lw=0.6, alpha=0.6, linestyle='--')
if t_onset:
    ax3.axvline(t_onset, color='yellow', lw=2.0, linestyle='--')
plt.colorbar(pcm, ax=ax3)
plt.tight_layout()
plt.savefig('q2_spectrogram.png', dpi=150)

t_t = np.arange(L) / fs * 1000
fig3, ax4 = plt.subplots(figsize=(8, 3))
ax4.plot(t_t, tmpl, color='#6366f1', lw=2)
ax4.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('q2_template.png', dpi=150)
