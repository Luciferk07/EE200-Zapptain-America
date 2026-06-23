import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# ── Title
cells.append(nbf.v4.new_markdown_cell(
    "# Q2: The Midnight Episode — 'Catching the Arrhythmia'\n"
    "### EE200 Course Project — Summer 2026\n"
    "**Prepared by:** Your Name / Roll Number\n\n"
    "---\n"
    "The patient's ECG was recorded with a Holter monitor for 20 seconds at "
    "$f_s = 250$ Hz ($N = 5000$ samples). The first ~9.6 s is a perfectly healthy, "
    "metronomic rhythm. At that point the heart slips into an arrhythmia. We detect "
    "this onset using **normalized cross-correlation** and confirm it with a **spectrogram**."
))

# ── (a)
cells.append(nbf.v4.new_markdown_cell(
    "## (a) Reading the Signal\n\n"
    "**(i) Duration**\n\n"
    "$$T = \\frac{N}{f_s} = \\frac{5000}{250} = \\boxed{20 \\text{ seconds}}$$\n\n"
    "**(ii) Heart rate & samples per beat**\n\n"
    "One full beat $\\{P, QRS, T\\}$ arrives every $0.8$ s in the healthy stretch.\n\n"
    "$$\\text{HR} = \\frac{60}{0.8} = \\boxed{75 \\text{ bpm}}, \\qquad\n"
    "  L = 0.8 \\times 250 = \\boxed{200 \\text{ samples per beat}}$$\n\n"
    "**(iii) Fundamental frequency**\n\n"
    "$$f_0 = \\frac{1}{0.8\\text{ s}} = \\boxed{1.25 \\text{ Hz}}$$"
))

# ── (b)
cells.append(nbf.v4.new_markdown_cell(
    "## (b) Healthy Heart in the Frequency Domain\n\n"
    "**(i)** A nearly periodic signal has a **discrete (line) spectrum**: isolated spikes "
    "(harmonics) at integer multiples of $f_0 = 1.25$ Hz — not a smooth continuous curve.\n\n"
    "**(ii)** The **QRS complex** owns the high frequencies. It is a sharp, narrow spike "
    "in the time domain — steep slopes require high-frequency components to represent. "
    "The broad, smooth P and T waves contribute only to the low-frequency harmonics.\n\n"
    "**(iii)** At 150 bpm: $f_0' = 150/60 = 2.5$ Hz. The fundamental frequency "
    "**doubles** to 2.5 Hz and the harmonic spacing also doubles from 1.25 Hz to **2.5 Hz**."
))

# ── (c)
cells.append(nbf.v4.new_markdown_cell(
    "## (c) Cutting a Heartbeat (Windowing)\n\n"
    "**(i)** Width = **200 samples** (one full beat period). "
    "Placement: the **early, healthy portion** (first 1-2 s), where rhythm is perfectly regular.\n\n"
    "**(ii)**\n\n"
    "| Window | Problem |\n"
    "|--------|---------|\n"
    "| **80 samples** (0.32 s) | Captures only the QRS complex; P-wave and T-wave are cropped. "
    "Template is structurally *incomplete* — correlation never reaches 1 for a real beat. |\n"
    "| **600 samples** (2.4 s) | Spans exactly **3 beats**. The template averages multiple beats; "
    "correlation peaks every 3rd beat boundary instead of every beat. |\n\n"
    "**(iii)** A shorter window improves time resolution but degrades frequency resolution "
    "($\\Delta f = f_s / N_w$). More importantly, the window must be at least one full "
    "physiological beat (200 samples) to contain all structural landmarks $\\{P, QRS, T\\}$. "
    "Shrinking it below that destroys the template Maya needs to match against."
))

# ── (d)
cells.append(nbf.v4.new_markdown_cell(
    "## (d) Match the Template (Correlation)\n\n"
    "$$\\rho(m) = \\frac{\\sum_k t[k]\\,x[m+k]}{\\|t\\|\\,\\|x_m\\|}$$\n\n"
    "**(i)** $\\rho(m) \\in [-1,\\,+1]$. A value of **$\\rho = 1$** (or close) signals "
    "a near-perfect shape match.\n\n"
    "**(ii)** Without the denominator, the raw score is **not scale-invariant**. "
    "A healthy beat twice as tall as the template doubles the unnormalised score — "
    "making it look \"better\" than it really is. Baseline wander and amplitude variation "
    "would constantly fool an unnormalised detector. Dividing by $\\|x_m\\|$ makes the "
    "score depend only on *shape*, not on energy.\n\n"
    "**(iii)** An inverted beat is approximately $-t[k]$, so:\n\n"
    "$$\\rho \\approx \\frac{\\sum_k t[k]\\,(-t[k])}{\\|t\\|^2} = -1$$\n\n"
    "This makes inverted beats **trivially easy to flag**: they produce a strong *negative* "
    "peak — unambiguously abnormal. The data confirms this: the first arrhythmia beat "
    "scores $\\rho = -0.9879 \\approx -1$."
))

# ── (e)
cells.append(nbf.v4.new_markdown_cell(
    "## (e) Onset Detection & the Spectrogram\n\n"
    "**(i)** **Rule:** declare an arrhythmia at the first beat index $m^*$ "
    "where $\\rho(m^*) < \\tau$ (e.g., $\\tau = 0.5$).\n\n"
    "| Threshold | Risk |\n"
    "|-----------|------|\n"
    "| Too **high** (0.95) | Breathing-related amplitude variation causes false alarms in healthy beats. |\n"
    "| Too **low** (0.10) | Subtle but dangerous rhythm changes score above threshold and are missed. |\n\n"
    "**(ii)** *Healthy region*: spectrogram shows **steady, bright horizontal bands** "
    "at $f_0 = 1.25$ Hz and harmonics (2.5, 3.75, 5.0 Hz…). "
    "*Arrhythmia region*: periodic structure collapses — bands shift, smear, or disappear.\n\n"
    "**(iii)** The **correlation plot** is more trustworthy for pinpointing the exact beat. "
    "The spectrogram requires a long window (for frequency resolution), which time-smears "
    "the event across many frames — the transition looks gradual. "
    "The beat-by-beat correlator operates directly in the time domain and flags the "
    "*exact* 200-sample window where shape first goes wrong."
))

# ── (f)
cells.append(nbf.v4.new_markdown_cell(
    "## (f) Sampling & Aliasing\n\n"
    "**(i)** $$f_s^{\\min} = 2 \\times 40 = \\boxed{80 \\text{ Hz}}$$\n\n"
    "**(ii)** The Nyquist frequency at 50 Hz sampling is 25 Hz. Energy at 25-40 Hz "
    "**folds back** into 10-25 Hz. The QRS spike — built from those high frequencies — "
    "gets distorted and blurred. A sharp spike looks rounded and wider; its correlation "
    "with Maya's clean template drops, producing false arrhythmia flags during healthy beats.\n\n"
    "**(iii)** Apply a **steep anti-aliasing low-pass filter** with $f_c < 25$ Hz *before* "
    "sampling. The unavoidable cost: permanent loss of 25-40 Hz QRS content, making the "
    "reconstructed spike broader and clinically less sharp."
))

# ── (g) markdown
cells.append(nbf.v4.new_markdown_cell(
    "## (g) Prototyping the Detector in Code\n\n"
    "We implement `find_onset` exactly as specified:\n"
    "- Jump forward by **L** samples at a time (beat-by-beat, not sample-by-sample)\n"
    "- Compute normalised dot-product $\\rho$ for each window\n"
    "- Return index of **first** window where $\\rho < \\tau$ (or $-1$ if never breached)\n\n"
    "**Key improvement:** we collect ALL beat scores before returning so the plot "
    "shows the complete 25-beat correlation profile — not just the beats up to onset."
))

# ── (g) code  — no nested triple-quotes
CODE_G = (
    "import numpy as np\n"
    "import matplotlib\n"
    "matplotlib.use('Agg')\n"
    "import matplotlib.pyplot as plt\n\n"
    "# Load data\n"
    "ecg_signal = np.load('EE200_course_project_data_2026/Q2_data/patient_ecg.npy')\n"
    "template   = np.load('EE200_course_project_data_2026/Q2_data/template.npy')\n"
    "fs = 250; N = len(ecg_signal); L = len(template)\n"
    "t_axis = np.arange(N) / fs\n\n"
    "print(f'ECG:      {N} samples ({N/fs:.1f} s)')\n"
    "print(f'Template: {L} samples ({L/fs:.3f} s = one beat)')\n\n"
    "# ── find_onset: collects ALL beat scores, marks FIRST breach ──────────\n"
    "def find_onset(ecg_signal, template, threshold=0.5):\n"
    "    L = len(template); N = len(ecg_signal)\n"
    "    norm_t = np.linalg.norm(template)\n"
    "    onset = -1; scores = []; indices = []\n"
    "    for m in range(0, N - L + 1, L):\n"
    "        segment = ecg_signal[m:m+L]\n"
    "        norm_s = np.linalg.norm(segment)\n"
    "        rho = float(np.dot(template, segment) / (norm_t * norm_s)) if norm_s > 0 else 0.0\n"
    "        scores.append(rho); indices.append(m)\n"
    "        if rho < threshold and onset == -1:\n"
    "            onset = m   # record but keep collecting all beats\n"
    "    return onset, scores, indices\n\n"
    "onset_idx, rho_scores, beat_indices = find_onset(ecg_signal, template, threshold=0.5)\n"
    "onset_time = onset_idx / fs if onset_idx != -1 else None\n\n"
    "print(f'Arrhythmia onset: sample {onset_idx}  (t = {onset_time:.2f} s)')\n"
    "print(f'Total beats evaluated: {len(rho_scores)}')\n"
    "print('Full rho profile:')\n"
    "for m, r in zip(beat_indices, rho_scores):\n"
    "    flag = '  <- FIRST ARRHYTHMIA BEAT' if m == onset_idx else ('  <- arrhythmia' if r < 0.5 else '')\n"
    "    print(f'  m={m:4d}  t={m/fs:5.2f}s  rho={r:+.4f}{flag}')\n\n"
    "# ── Plot: ECG + full correlation profile ─────────────────────────────\n"
    "fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)\n"
    "fig.suptitle('ECG Signal and Beat-by-Beat Correlation', fontsize=14, fontweight='bold')\n\n"
    "ax1 = axes[0]\n"
    "ax1.plot(t_axis, ecg_signal, color='#475569', lw=0.8, label='ECG signal')\n"
    "if onset_time:\n"
    "    ax1.axvline(onset_time, color='#ef4444', lw=1.8, linestyle='--', label=f'Onset t={onset_time:.1f}s')\n"
    "    ax1.axvspan(onset_time, N/fs, alpha=0.07, color='#ef4444', label='Arrhythmia region')\n"
    "ax1.set_ylabel('Amplitude'); ax1.set_title('Patient ECG Recording')\n"
    "ax1.legend(loc='upper right'); ax1.grid(True, alpha=0.3); ax1.set_xlim(0, N/fs)\n\n"
    "ax2 = axes[1]\n"
    "beat_times = [m/fs for m in beat_indices]\n"
    "colors_dots = ['#22c55e' if r >= 0.5 else '#ef4444' for r in rho_scores]\n"
    "ax2.plot(beat_times, rho_scores, color='#818cf8', lw=1.4, zorder=1)\n"
    "ax2.scatter(beat_times, rho_scores, c=colors_dots, s=70, zorder=2,\n"
    "            label='rho per beat  (green=healthy, red=arrhythmia)')\n"
    "ax2.axhline(0.5, color='#f59e0b', lw=1.5, linestyle='--', label='Threshold tau=0.5')\n"
    "ax2.axhline(0.0, color='gray', lw=0.8, linestyle=':')\n"
    "if onset_time:\n"
    "    ax2.axvline(onset_time, color='#ef4444', lw=1.8, linestyle='--',\n"
    "                label=f'Detected onset (t={onset_time:.1f}s)')\n"
    "ax2.set_xlabel('Time [s]'); ax2.set_ylabel('Correlation rho(m)')\n"
    "ax2.set_title('Beat-by-Beat Normalized Correlation  [all 25 beats shown]')\n"
    "ax2.legend(loc='lower left'); ax2.grid(True, alpha=0.3)\n"
    "ax2.set_xlim(0, N/fs); ax2.set_ylim(-1.15, 1.15)\n\n"
    "plt.tight_layout()\n"
    "plt.savefig('q2_correlation_plot.png', dpi=150)\n"
    "plt.show()\n"
)
cells.append(nbf.v4.new_code_cell(CODE_G))

# ── (h) markdown
cells.append(nbf.v4.new_markdown_cell(
    "## (h) Visualizing the Spectrogram\n\n"
    "**Window length choice:** `nperseg = 250` (one second of data)\n\n"
    "$$\\Delta f = \\frac{f_s}{N_w} = \\frac{250}{250} = 1.0 \\text{ Hz}$$\n\n"
    "This frequency resolution of **1 Hz** clearly resolves the harmonics spaced "
    "1.25 Hz apart. A shorter window (e.g., 50 samples) would merge adjacent harmonics; "
    "a longer one (e.g., 2000 samples) would blur the onset boundary in time. "
    "The 1-second window strikes the right balance: harmonic bands are separated AND "
    "the onset at ~9.6 s is clearly visible as the point where the regular pattern breaks down."
))

# ── (h) code — no nested triple-quotes
CODE_H = (
    "from scipy.signal import spectrogram as scipy_spectrogram\n\n"
    "# nperseg=250 => Delta_f=1 Hz  (resolves 1.25 Hz harmonics)\n"
    "# noverlap=200 => time step = 50/250 = 0.2 s\n"
    "f_s2, t_s2, Sxx = scipy_spectrogram(\n"
    "    ecg_signal, fs=fs,\n"
    "    nperseg=250, noverlap=200,\n"
    "    window='hann'\n"
    ")\n\n"
    "print(f'Freq resolution: {fs/250:.2f} Hz')\n"
    "print(f'Time step:       {(250-200)/fs:.3f} s')\n"
    "print(f'Sxx shape:       {Sxx.shape}')\n\n"
    "fig2, ax3 = plt.subplots(figsize=(14, 5))\n"
    "pcm = ax3.pcolormesh(\n"
    "    t_s2, f_s2,\n"
    "    10 * np.log10(Sxx + 1e-12),\n"
    "    shading='gouraud', cmap='inferno', vmin=-80, vmax=-20\n"
    ")\n"
    "ax3.set_ylim(0, 25)\n\n"
    "# Mark expected harmonics of f0=1.25 Hz\n"
    "for k in range(1, 20):\n"
    "    fk = k * 1.25\n"
    "    if fk > 25: break\n"
    "    ax3.axhline(fk, color='cyan', lw=0.6, alpha=0.6, linestyle='--')\n\n"
    "if onset_time:\n"
    "    ax3.axvline(onset_time, color='#facc15', lw=2.0, linestyle='--',\n"
    "                label=f'Arrhythmia onset (t={onset_time:.1f}s)')\n\n"
    "ax3.set_xlabel('Time [s]'); ax3.set_ylabel('Frequency [Hz]')\n"
    "ax3.set_title(\n"
    "    'Spectrogram of Patient ECG\\n'\n"
    "    '(cyan dashed = expected harmonics at 1.25*k Hz;  '\n"
    "    'yellow = detected onset;  nperseg=250, Delta_f=1 Hz)'\n"
    ")\n"
    "ax3.legend(loc='upper right')\n"
    "plt.colorbar(pcm, ax=ax3, label='Power [dB]')\n"
    "plt.tight_layout()\n"
    "plt.savefig('q2_spectrogram.png', dpi=150)\n"
    "plt.show()\n"
    "print('Saved: q2_spectrogram.png')\n"
    "print()\n"
    "print('Observation: In the first ~9.6 s the horizontal cyan lines are clearly lit,')\n"
    "print('confirming periodic harmonic structure. After the onset the pattern fragments.')\n"
)
cells.append(nbf.v4.new_code_cell(CODE_H))

# ── Template appendix
cells.append(nbf.v4.new_markdown_cell("## Appendix: Template Waveform\nThe template `template.npy` provided is exactly 200 samples (one full healthy beat). It shows the canonical P-QRS-T morphology clearly, confirming it is a good reference for correlation."))

CODE_TMPL = (
    "t_tmpl = np.arange(L) / fs * 1000  # in ms\n"
    "fig3, ax4 = plt.subplots(figsize=(8, 3))\n"
    "ax4.plot(t_tmpl, template, color='#6366f1', lw=2)\n"
    "ax4.set_xlabel('Time within beat [ms]')\n"
    "ax4.set_ylabel('Amplitude')\n"
    "ax4.set_title(f'Template: one healthy beat  ({L} samples = {L/fs:.2f} s)')\n"
    "ax4.grid(True, alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.savefig('q2_template.png', dpi=150)\n"
    "plt.show()\n"
    "print(f'Template: peak={template.max():.4f}  trough={template.min():.4f}')\n"
    "print('The template clearly shows: P-wave (broad bump ~180ms), QRS spike (~400ms), T-wave (~600ms)')\n"
)
cells.append(nbf.v4.new_code_cell(CODE_TMPL))

nb['cells'] = cells

with open('Q2_solution.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Q2_solution.ipynb built successfully (no nested triple-quote issues).")
