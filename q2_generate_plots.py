import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import spectrogram as scipy_spectrogram

# ── Load data ────────────────────────────────────────────────────────────
ecg_signal = np.load('EE200_course_project_data_2026/Q2_data/patient_ecg.npy')
template   = np.load('EE200_course_project_data_2026/Q2_data/template.npy')

fs = 250          # sampling frequency [Hz]
N  = len(ecg_signal)
L  = len(template)
t_axis = np.arange(N) / fs

print(f"ECG:      {N} samples  ({N/fs:.1f} s)")
print(f"Template: {L} samples  ({L/fs:.3f} s = one beat)")

# ─────────────────────────────────────────────────────────────────────────
def find_onset(ecg_signal, template, threshold=0.5):
    # Beat-by-beat normalized cross-correlation detector.
    # Jumps forward by L = len(template) samples (one beat at a time).
    # Collects ALL beat scores first, then identifies the first breach.
    # Returns: (onset_index, all_scores, all_indices)
    L      = len(template)
    N      = len(ecg_signal)
    norm_t = np.linalg.norm(template)
    onset  = -1
    scores, indices = [], []

    for m in range(0, N - L + 1, L):
        segment = ecg_signal[m : m + L]
        norm_s  = np.linalg.norm(segment)
        rho = float(np.dot(template, segment) / (norm_t * norm_s)) if norm_s > 0 else 0.0
        scores.append(rho)
        indices.append(m)
        if rho < threshold and onset == -1:
            onset = m   # mark first breach but keep collecting

    return onset, scores, indices
# ─────────────────────────────────────────────────────────────────────────

onset_idx, rho_scores, beat_indices = find_onset(ecg_signal, template, threshold=0.5)
onset_time = onset_idx / fs if onset_idx != -1 else None

print(f"\nArrhythmia onset: sample {onset_idx}  (t = {onset_time:.2f} s)")
print(f"Total beats evaluated: {len(rho_scores)}")
print(f"\nFull rho profile:")
for m, r in zip(beat_indices, rho_scores):
    flag = "  <- ARRHYTHMIA (first)" if m == onset_idx else ("  <- arrhythmia" if r < 0.5 else "")
    print(f"  m={m:4d}  t={m/fs:5.2f}s  rho={r:+.4f}{flag}")

# ── Figure 1: ECG + full correlation profile ─────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
fig.suptitle("ECG Signal and Beat-by-Beat Correlation", fontsize=14, fontweight='bold')

# Top panel: raw ECG
ax1 = axes[0]
ax1.plot(t_axis, ecg_signal, color='#475569', lw=0.8, label='ECG signal')
if onset_time:
    ax1.axvline(onset_time, color='#ef4444', lw=1.8, linestyle='--',
                label=f'Onset t={onset_time:.1f}s')
    ax1.axvspan(onset_time, N/fs, alpha=0.07, color='#ef4444', label='Arrhythmia region')
ax1.set_ylabel('Amplitude')
ax1.set_title('Patient ECG Recording')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, N/fs)

# Bottom panel: full correlation profile
ax2 = axes[1]
beat_times = [m/fs for m in beat_indices]
colors = ['#22c55e' if r >= 0.5 else '#ef4444' for r in rho_scores]
ax2.plot(beat_times, rho_scores, color='#818cf8', lw=1.4, zorder=1)
ax2.scatter(beat_times, rho_scores, c=colors, s=70, zorder=2,
            label='rho per beat  (green=healthy, red=arrhythmia)')
ax2.axhline(0.5,  color='#f59e0b', lw=1.5, linestyle='--', label='Threshold tau = 0.5')
ax2.axhline(0.0,  color='gray',    lw=0.8, linestyle=':')
if onset_time:
    ax2.axvline(onset_time, color='#ef4444', lw=1.8, linestyle='--',
                label=f'Detected onset (t={onset_time:.1f}s)')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Correlation rho(m)')
ax2.set_title('Beat-by-Beat Normalized Correlation  [all 25 beats shown]')
ax2.legend(loc='lower left')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, N/fs)
ax2.set_ylim(-1.15, 1.15)

plt.tight_layout()
plt.savefig('q2_correlation_plot.png', dpi=150)
plt.show()
print("Saved: q2_correlation_plot.png")

# ── Figure 2: Spectrogram ────────────────────────────────────────────────
# nperseg=250 => Delta_f = 250/250 = 1.0 Hz  (clearly resolves 1.25 Hz harmonics)
# noverlap=200 => time step = 50/250 = 0.2 s
f_spect, t_spect, Sxx = scipy_spectrogram(
    ecg_signal, fs=fs,
    nperseg=250, noverlap=200,
    window='hann'
)

print(f"\nSpectrogram freq resolution: {fs/250:.2f} Hz")
print(f"Spectrogram time step:       {(250-200)/fs:.3f} s")
print(f"Sxx shape:                   {Sxx.shape}  (freq_bins x time_frames)")

fig2, ax3 = plt.subplots(figsize=(14, 5))
pcm = ax3.pcolormesh(
    t_spect, f_spect,
    10 * np.log10(Sxx + 1e-12),
    shading='gouraud',
    cmap='inferno',
    vmin=-80, vmax=-20
)
ax3.set_ylim(0, 25)

# Mark expected harmonics of f0=1.25 Hz
for k in range(1, 20):
    fk = k * 1.25
    if fk > 25:
        break
    ax3.axhline(fk, color='cyan', lw=0.6, alpha=0.6, linestyle='--')

if onset_time:
    ax3.axvline(onset_time, color='#facc15', lw=2.0, linestyle='--',
                label=f'Arrhythmia onset (t={onset_time:.1f}s)')

ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Frequency [Hz]')
ax3.set_title(
    'Spectrogram of Patient ECG\n'
    '(cyan dashed = expected harmonics at 1.25 Hz x k;  '
    'yellow = detected onset;  nperseg=250, Delta_f=1 Hz)'
)
ax3.legend(loc='upper right')
plt.colorbar(pcm, ax=ax3, label='Power [dB]')
plt.tight_layout()
plt.savefig('q2_spectrogram.png', dpi=150)
plt.show()
print("Saved: q2_spectrogram.png")

# ── Figure 3: Template waveform ──────────────────────────────────────────
t_tmpl = np.arange(L) / fs * 1000  # ms

fig3, ax4 = plt.subplots(figsize=(8, 3))
ax4.plot(t_tmpl, template, color='#6366f1', lw=2)
ax4.set_xlabel('Time within beat [ms]')
ax4.set_ylabel('Amplitude')
ax4.set_title(f'Template: one healthy beat  ({L} samples = {L/fs:.2f} s)')
ax4.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('q2_template.png', dpi=150)
plt.show()
print("Saved: q2_template.png")
print("\nAll Q2 plots generated successfully.")
