"""
FigS1: Prior Comparison on TCGA-BRCA (d=300, real experiment).
(a) Prior vs CAGate edge counts
(b) Novel target identification matrix
"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(PKG_ROOT, 'figures_supplementary')
os.makedirs(OUT_DIR, exist_ok=True)

CB = {'blue':'#0072B2','verm':'#D55E00','green':'#009E73','dark':'#333333','gray':'#999999'}

plt.rcParams.update({
    'font.family':'serif','font.serif':['DejaVu Serif','Times New Roman'],
    'font.size':8,'axes.labelsize':9,
    'xtick.labelsize':7.5,'ytick.labelsize':7,'legend.fontsize':7,
    'axes.linewidth':0.6,'axes.spines.top':False,'axes.spines.right':False,
    'axes.labelpad':2,'figure.facecolor':'white','axes.facecolor':'white',
    'savefig.dpi':400,'savefig.bbox':'tight','savefig.pad_inches':0.05,
})

def plabel(ax, t):
    ax.text(-0.08, 1.06, t, transform=ax.transAxes, fontsize=13,
            fontweight='bold', va='bottom', ha='left', color=CB['dark'])

# === REAL DATA ===
labels = ['Weak (all-to-all)', 'ENCODE (curated)']
prior_vals = [14950, 429]
cagate_vals = [67, 68]
targets = ['PGR', 'ERBB4', 'SCUBE2']
target_matrix = np.array([[1, 0, 1], [0, 1, 1]])

fig = plt.figure(figsize=(8.5, 4.2))
gs = fig.add_gridspec(1, 2, wspace=0.50, left=0.08, right=0.98, bottom=0.14, top=0.92)

# (a) Prior vs CAGate: grouped bar, log scale
ax = fig.add_subplot(gs[0, 0])
x = np.arange(2)
w = 0.32

b1 = ax.bar(x - w/2, prior_vals, w, color=CB['gray'], alpha=0.65,
            label='Prior edges', edgecolor='white', linewidth=0.3)
b2 = ax.bar(x + w/2, cagate_vals, w, color=CB['blue'], alpha=0.92,
            label='CAGate edges', edgecolor='white', linewidth=0.3)
ax.set_yscale('log')
ax.set_ylabel('Edge count (log scale)')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)

# Value labels on log scale
for bar, val in zip(b1, prior_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val * 1.35,
            format(val, ','), ha='center', fontsize=7.5, weight='bold',
            color=CB['gray'], va='bottom')
for bar, val in zip(b2, cagate_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val * 4.5,
            str(val), ha='center', fontsize=8, weight='bold',
            color=CB['blue'], va='bottom')

# Enrichment callout - top left
ax.text(0.04, 0.83, 'Retention: 0.45% vs 16%\n35x sparser, same output',
        transform=ax.transAxes, ha='left', va='top', fontsize=7, color=CB['blue'],
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=CB['gray'], alpha=0.92, lw=0.3))

ax.legend(fontsize=7, loc='upper right', framealpha=0.85,
          bbox_to_anchor=(0.97, 0.97))
ax.set_ylim(8, 25000)
plabel(ax, 'a')

# (b) Target identification matrix
ax = fig.add_subplot(gs[0, 1])
ax.imshow(target_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1, alpha=0.88)

# Cell borders
for i in [-0.5, 0.5, 1.5, 2.5]:
    ax.axvline(i, color='white', lw=0.8)
for j in [-0.5, 0.5, 1.5]:
    ax.axhline(j, color='white', lw=0.8)

for i in range(3):
    for j in range(2):
        if target_matrix[j, i] == 1:
            ax.text(i, j, 'Found', ha='center', va='center', fontsize=9,
                    fontweight='bold', color=CB['dark'])
        else:
            ax.text(i, j, '-', ha='center', va='center', fontsize=14,
                    color=CB['gray'], alpha=0.5)

ax.set_xticks(range(3))
ax.set_xticklabels(targets, fontsize=9)
ax.set_yticks(range(2))
ax.set_yticklabels(['Weak (all-to-all)', 'ENCODE (curated)'], fontsize=9)

# SCUBE2 callout
ax.annotate('SCUBE2: found\nunder both priors', xy=(2, 0.5),
            xytext=(2.2, -0.7), fontsize=6.5, ha='center', color=CB['green'],
            weight='bold',
            arrowprops=dict(arrowstyle='->', color=CB['green'], lw=0.6,
                          connectionstyle='arc3,rad=-0.2'))

legend_elements = [
    Patch(facecolor=CB['blue'], alpha=0.85, label='Identified by CAGate'),
    Patch(facecolor='white', edgecolor=CB['gray'], label='Not identified')]
ax.legend(handles=legend_elements, fontsize=7, loc='upper left', framealpha=0.85)
plabel(ax, 'b')

out = os.path.join(OUT_DIR, 'FigS1_Supplementary.pdf')
fig.savefig(out, dpi=400, facecolor='white', edgecolor='none')
plt.close(fig)
print('[SAVED] FigS1_Supplementary.pdf (%d KB)' % (os.path.getsize(out) // 1024))
