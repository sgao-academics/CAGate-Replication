"""FigS2: Prior comparison for CAGate on TCGA-BRCA (300-gene subset).
(a) Prior vs CAGate-learned edges; (b) Novel target gene identification.
Adapted from cancer_application/scripts/generate_figures.py with CAGate labeling.
"""
import os
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.unicode_minus'] = False

FIG_DIR = os.path.join(PKG_ROOT, 'figures_supplementary')
os.makedirs(FIG_DIR, exist_ok=True)

N_BLUE, N_ORANGE, N_GRAY, N_DARK = '#3C5488', '#F39C12', '#BDBDBD', '#2C3E50'

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# ── (a) Grouped bar: prior vs CAGate edges, log Y ──
ax = axes[0]
labels = ['Weak\n(all-to-all)', 'ENCODE\n(curated+coexp)']
prior_vals = [14950, 429]
cagate_vals = [68, 67]
x = np.arange(len(labels))
w = 0.3

bars1 = ax.bar(x - w/2, prior_vals, w, label='Prior edges', color=N_GRAY, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + w/2, cagate_vals, w, label='CAGate-learned edges', color=N_BLUE, edgecolor='black', linewidth=0.5)
ax.set_ylabel('Number of Edges (log scale)')
ax.set_title('(a) Edge Counts: Prior vs Learned')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_yscale('log')
ax.legend(fontsize=8, loc='upper right')

for bar, val in zip(bars1, prior_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
            f'{val:,}', ha='center', fontsize=8, fontweight='bold')
for bar, val in zip(bars2, cagate_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.8,
            str(val), ha='center', fontsize=9, fontweight='bold', color=N_BLUE)

# ── (b) Presence matrix heatmap ──
ax = axes[1]
targets = ['PGR', 'ERBB4', 'SCUBE2']
prior_types = ['Weak', 'ENCODE']
data = np.array([[1, 0, 0], [0, 1, 1]])
im = ax.imshow(data, cmap='Blues', aspect='auto', vmin=0, vmax=1, alpha=0.8)
for i in range(3):
    for j in range(2):
        if data[j, i] == 1:
            ax.text(i, j, 'Yes', ha='center', va='center', fontsize=10,
                    fontweight='bold', color=N_DARK)
        else:
            ax.text(i, j, '--', ha='center', va='center', fontsize=12, color=N_GRAY)
ax.set_xticks(range(len(targets)))
ax.set_xticklabels(targets, fontsize=10)
ax.set_yticks(range(len(prior_types)))
ax.set_yticklabels(prior_types, fontsize=10)
ax.set_title('(b) Novel Targets Identified')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=N_BLUE, alpha=0.8, label='Identified'),
                   Patch(facecolor='white', edgecolor=N_GRAY, label='Not identified')]
ax.legend(handles=legend_elements, fontsize=8, loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'FigS2_PriorComparison.png'), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'FigS2_PriorComparison.pdf'), bbox_inches='tight')
plt.close()
print('FigS2 saved')
