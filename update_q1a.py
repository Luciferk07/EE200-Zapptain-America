import json

with open('Q1A_solution.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update cell 5 (Detecting Interference Spikes)
nb['cells'][5]['source'] = [
    "thresh = np.mean(mag_db) + 5 * np.std(mag_db)\n",
    "spike_coords = np.argwhere(mag_db > thresh)\n",
    "\n",
    "# Exclude the DC component at the center\n",
    "center_r, center_c = mag_db.shape[0] // 2, mag_db.shape[1] // 2\n",
    "peaks = [(r, c) for r, c in spike_coords if np.sqrt((r - center_r)**2 + (c - center_c)**2) > 20]\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(12, 7))\n",
    "ax.imshow(mag_db, cmap='hot')\n",
    "for (r, c) in peaks:\n",
    "    ax.add_patch(plt.Circle((c, r), 8, color='cyan', fill=False, lw=1.5))\n",
    "ax.set_title('Annotated Spectrum — Interference Spikes')\n",
    "ax.axis('off')\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{OUT}/03_annotated_spectrum.png', bbox_inches='tight')\n",
    "plt.show()\n",
    "\n",
    "print(f'Detected {len(peaks)} spike pixels')"
]

with open('Q1A_solution.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Updated Q1A_solution.ipynb successfully.")
