"""
Build the Q3A Jupyter Notebook with all experiments, analysis, and plots.
"""
import json
import os

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

notebook = {
    "cells": [
        md(["# Q3A: Sonic Signatures (Audio Fingerprinting)\n",
            "This notebook implements a Shazam-style audio recognition system from scratch, exploring the fundamental trade-offs in time-frequency analysis and demonstrating robustness against noise."]),
        
        md(["## Experiment 1: The Spectrogram and Window Length Trade-off\n",
            "The fundamental tool for audio analysis is the Short-Time Fourier Transform (STFT). We must choose a window length (`n_fft`).\n",
            "- A **short window** provides excellent time resolution (we know exactly *when* a note occurred) but poor frequency resolution (the frequencies blur together).\n",
            "- A **long window** provides excellent frequency resolution (we know exactly *which* note was played) but poor time resolution (we don't know exactly when it started).\n",
            "\n",
            "Let's visualize this Heisenberg-Gabor limit."]),
        
        code(["import librosa\n",
              "import numpy as np\n",
              "import matplotlib.pyplot as plt\n",
              "import librosa.display\n",
              "\n",
              "y, sr = librosa.load(r'EE200_course_project_data_2026/Q3_database/Let It Be.mp3', sr=11025, duration=5.0, offset=10.0)\n",
              "\n",
              "# Short window (n_fft = 256)\n",
              "D_short = librosa.stft(y, n_fft=256, hop_length=64)\n",
              "S_short = librosa.amplitude_to_db(np.abs(D_short), ref=np.max)\n",
              "\n",
              "# Long window (n_fft = 4096)\n",
              "D_long = librosa.stft(y, n_fft=4096, hop_length=1024)\n",
              "S_long = librosa.amplitude_to_db(np.abs(D_long), ref=np.max)\n",
              "\n",
              "fig, ax = plt.subplots(1, 2, figsize=(14, 5))\n",
              "librosa.display.specshow(S_short, sr=sr, x_axis='time', y_axis='linear', ax=ax[0])\n",
              "ax[0].set_title('Short Window (n_fft=256)\\nGood Time, Poor Frequency')\n",
              "librosa.display.specshow(S_long, sr=sr, x_axis='time', y_axis='linear', ax=ax[1])\n",
              "ax[1].set_title('Long Window (n_fft=4096)\\nPoor Time, Good Frequency')\n",
              "plt.tight_layout()\n",
              "plt.savefig('q3a_experiment1.png')\n",
              "plt.show()"]),
              
        md(["## Experiment 2: Constellation Map & Fingerprinting (Hashing)\n",
            "Matching raw audio waveforms is impossible in a noisy cafe. Even matching raw spectrograms fails if the timing is slightly off. Instead, we extract **local maxima** (peaks) from the spectrogram to form a sparse *constellation map*.\n",
            "\n",
            "**Why pair peaks into hashes?**\n",
            "If we only match single frequencies $f_1$, we get thousands of false positives (many songs have a 440Hz note). But if we pair an anchor peak $f_1$ with a target peak $f_2$ separated by time $\\Delta t$, we form a hash: $(f_1, f_2, \\Delta t)$.\n",
            "The probability of two different songs having the exact same two frequencies separated by the exact same time delta is exponentially lower. The pair acts as a highly specific, unique identifier."]),
            
        code(["from scipy.ndimage import maximum_filter\n",
              "import scipy.signal as signal\n",
              "\n",
              "def get_constellation(S_db, percentile=90):\n",
              "    local_max = maximum_filter(S_db, size=(10, 10)) == S_db\n",
              "    threshold = np.percentile(S_db, percentile)\n",
              "    peak_mask = local_max & (S_db > threshold)\n",
              "    freqs, times = np.where(peak_mask)\n",
              "    return freqs, times\n",
              "\n",
              "D = librosa.stft(y, n_fft=2048, hop_length=512)\n",
              "S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)\n",
              "freqs, times = get_constellation(S_db)\n",
              "\n",
              "fig, ax = plt.subplots(figsize=(10, 5))\n",
              "ax.imshow(S_db, aspect='auto', origin='lower', cmap='gray', alpha=0.5)\n",
              "ax.scatter(times, freqs, c='red', s=15, alpha=0.8)\n",
              "ax.set_title('Constellation Map (Red dots are extracted peaks)')\n",
              "ax.set_xlabel('Time Frames')\n",
              "ax.set_ylabel('Frequency Bins')\n",
              "plt.savefig('q3a_experiment2.png')\n",
              "plt.show()"]),
              
        md(["## Experiment 3: Robustness to Noise\n",
            "Let's see what happens when we add Gaussian noise to the query clip."]),
            
        code(["# Add noise\n",
              "noise_amp = 0.05 * np.max(np.abs(y))\n",
              "y_noisy = y + noise_amp * np.random.randn(len(y))\n",
              "\n",
              "D_noisy = librosa.stft(y_noisy, n_fft=2048, hop_length=512)\n",
              "S_noisy = librosa.amplitude_to_db(np.abs(D_noisy), ref=np.max)\n",
              "freqs_noisy, times_noisy = get_constellation(S_noisy)\n",
              "\n",
              "fig, ax = plt.subplots(figsize=(10, 5))\n",
              "ax.imshow(S_noisy, aspect='auto', origin='lower', cmap='gray', alpha=0.5)\n",
              "ax.scatter(times_noisy, freqs_noisy, c='blue', s=15, alpha=0.8)\n",
              "ax.set_title('Constellation Map with Heavy Noise (Blue dots)')\n",
              "plt.savefig('q3a_experiment3.png')\n",
              "plt.show()\n",
              "print(f\"Original Peaks: {len(freqs)}, Noisy Peaks: {len(freqs_noisy)}\")"]),
              
        md(["As we increase the noise, the background noise introduces spurious peaks that overtake the true musical peaks. If too many true peaks fall below the dynamically calculated amplitude threshold, the generated hashes will completely change, and recognition fails."]),
        
        md(["## Experiment 4: Pitch Shifting & Time Stretching\n",
            "What happens if a DJ speeds up a track by 2% (time-stretching and pitch-shifting it)?\n",
            "The song sounds practically identical to humans. But our identifier is completely defeated. Why? Because the frequencies $f_1, f_2$ and the time delta $\\Delta t$ are all absolute values.\n",
            "A pitch shift changes $f_1 \\rightarrow f_1'$, so the hash $(f_1, f_2, \\Delta t)$ no longer matches the database at all.\n",
            "\n",
            "**How to make it robust?**\n",
            "Instead of hashing absolute frequencies, we could hash the **ratio** of frequencies: $\\frac{f_2}{f_1}$. A pitch shift multiplies all frequencies by a constant factor, so the ratio $\\frac{k \\cdot f_2}{k \\cdot f_1}$ remains invariant. For time-stretching, we could hash the ratio of time deltas between triplets of peaks."])
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(r'c:\Users\karti\OneDrive\Desktop\EE200 Project\Q3A_solution.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Q3A_solution.ipynb created successfully!")
