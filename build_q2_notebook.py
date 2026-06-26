import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# Q2: The Midnight Episode\n"
    "### EE200 Course Project\n"
    "**Name:** Kartik Agrawal"
))

cells.append(nbf.v4.new_markdown_cell(
    "## (a) Reading the Signal\n"
    "**Total Duration:** $T = 5000 / 250 = 20$ s.\n\n"
    "**Heart Rate:** $60 / 0.8 = 75$ bpm.\n\n"
    "**Samples per beat:** $0.8 \\times 250 = 200$ samples.\n\n"
    "**Fundamental Frequency:** $f_0 = 1 / 0.8 = 1.25$ Hz."
))

cells.append(nbf.v4.new_markdown_cell(
    "## (b) Frequency Domain\n"
    "- The spectrum consists of discrete harmonic spikes at multiples of 1.25 Hz.\n"
    "- The QRS complex contributes to high frequencies.\n"
    "- If HR doubles to 150 bpm, $f_0$ becomes 2.5 Hz and harmonic spacing doubles."
))

cells.append(nbf.v4.new_markdown_cell(
    "## (c) Windowing\n"
    "- A 200-sample window captures exactly one beat.\n"
    "- An 80-sample window truncates physiological features.\n"
    "- A 600-sample window averages multiple beats."
))

cells.append(nbf.v4.new_markdown_cell(
    "## (d) Correlation\n"
    "- $\\rho(m) \\in [-1, 1]$.\n"
    "- Normalization ensures scale invariance against amplitude variations.\n"
    "- An inverted beat yields $\\rho \\approx -1$."
))

cells.append(nbf.v4.new_markdown_cell(
    "## (e) Onset & Spectrogram\n"
    "- $\\rho < \\tau$ flags onset. Too high = false alarms, too low = missed events.\n"
    "- Healthy region shows clear horizontal bands; arrhythmia shows chaotic noise.\n"
    "- Correlation is precise; spectrograms suffer from time-smearing."
))

cells.append(nbf.v4.new_markdown_cell(
    "## (f) Sampling & Aliasing\n"
    "- Nyquist minimum rate is $f_s \\ge 2 \\times 40 = 80$ Hz.\n"
    "- Sampling at 50 Hz causes aliasing, distorting QRS spikes.\n"
    "- Fix requires an analog anti-aliasing low-pass filter ($f_c < 25$ Hz)."
))

cells.append(nbf.v4.new_markdown_cell("## (g) Arrhythmia Detector implementation"))
CODE_G = (
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n\n"
    "ecg = np.load('EE200_course_project_data_2026/Q2_data/patient_ecg.npy')\n"
    "tmpl = np.load('EE200_course_project_data_2026/Q2_data/template.npy')\n\n"
    "def get_onset(ecg, tmpl, thr=0.5):\n"
    "    L, N = len(tmpl), len(ecg)\n"
    "    nt = np.linalg.norm(tmpl)\n"
    "    for m in range(0, N - L + 1, L):\n"
    "        seg = ecg[m : m+L]\n"
    "        ns = np.linalg.norm(seg)\n"
    "        if ns > 0:\n"
    "            rho = np.dot(tmpl, seg) / (nt * ns)\n"
    "            if rho < thr: return m\n"
    "    return -1\n\n"
    "onset = get_onset(ecg, tmpl, 0.5)\n"
    "print(f'Onset found at sample {onset} (t = {onset/250:.2f} s)')"
)
cells.append(nbf.v4.new_code_cell(CODE_G))

cells.append(nbf.v4.new_markdown_cell("## (h) Spectrogram generation"))
CODE_H = (
    "from scipy.signal import spectrogram\n\n"
    "f, t, Sxx = spectrogram(ecg, fs=250, nperseg=200, noverlap=150, window='hann')\n"
    "plt.figure(figsize=(12, 4))\n"
    "plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='inferno')\n"
    "plt.ylim(0, 25)\n"
    "if onset != -1: plt.axvline(onset/250, color='yellow', linestyle='--')\n"
    "plt.show()"
)
cells.append(nbf.v4.new_code_cell(CODE_H))

nb['cells'] = cells
with open('Q2_solution.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
