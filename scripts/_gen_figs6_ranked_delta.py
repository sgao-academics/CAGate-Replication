"""FigS6: CAGate delta distribution across 33 TCGA cancer types at d=100.
Data: mega_33_full.json (real experimental values).
(a) Boxplot + swarm; (b) Ranked horizontal bar chart.
Colorblind-friendly: blue/purple/teal palette.
"""
import os
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.unicode_minus'] = False

DATA = os.path.join(PKG_ROOT, 'data')
FIG_DIR = os.path.join(PKG_ROOT, 'figures_supplementary')
os.makedirs(FIG_DIR, exist_ok=True)

mega33 = json.load(open(os.path.join(DATA, 'mega_33_full.json')))

rows = []
for k, v in mega33.items():
    n = v.get('n', 0)
    delta = v.get('cagate_delta', 0)
    cag = v.get('cagate_mean', 0)
    nte = v.get('base_mean', 0)
    if delta <= 0:
        cag = v.get('cagate_mean', v.get('sscagate_mean', 0))
        nte = v.get('base_mean', 0)
        delta = cag - nte
    rows.append((k, n, delta, cag, nte))

rows.sort(key=lambda x: x[2], reverse=True)
codes = [r[0] for r in rows]
deltas = np.array([r[2] for r in rows])
mean_d = np.mean(deltas)
median_d = np.median(deltas)
print(f'FigS6: {len(rows)} cancers, mean={mean_d:.0f}, median={median_d:.0f}')

# Colorblind-friendly: blue (moderate), purple (large), teal (massive)
C_BLUE   = '#3C5488'  # moderate delta
C_PURPLE = '#7E57C2'  # large delta
C_TEAL   = '#00A087'  # massive delta
C_RED    = '#E64B35'  # accent only (mean line)
C_GRAY   = '#95A5A6'
C_DARK   = '#2C3E50'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

# ── (a) Boxplot + swarm ──
cs = [C_TEAL if d > 200 else C_PURPLE if d > 100 else C_BLUE for d in deltas]
bp = ax1.boxplot(deltas, vert=True, patch_artist=True, widths=0.3,
                 boxprops=dict(facecolor=C_BLUE, alpha=0.20, linewidth=0.8),
                 medianprops=dict(color=C_DARK, linewidth=2),
                 whiskerprops=dict(color=C_GRAY, linewidth=0.6),
                 capprops=dict(color=C_GRAY, linewidth=0.6))
jitter = np.random.uniform(-0.12, 0.12, len(deltas))
ax1.scatter(np.ones(len(deltas)) + jitter, deltas, c=cs, s=55,
            edgecolors='white', linewidth=0.3, zorder=3, alpha=0.85)
ax1.axhline(y=mean_d, color=C_RED, linestyle='--', alpha=0.6, linewidth=1, label=f'Mean = +{mean_d:.0f}')
ax1.axhline(y=median_d, color=C_PURPLE, linestyle=':', alpha=0.5, linewidth=1, label=f'Median = +{median_d:.0f}')
ax1.set_ylabel(r'$\Delta$ (CAGate $-$ NOTEARS, edges)')
ax1.set_title(f'(a) Delta Distribution ({len(rows)} Cancers)', fontsize=11, fontweight='bold')
ax1.legend(fontsize=7, loc='upper right', framealpha=0.85)
ax1.set_xticks([])
ax1.text(0.02, 1.02, 'a', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='bottom')

# ── (b) Ranked horizontal bar chart ──
idx = np.argsort(deltas)
sorted_codes = [codes[i] for i in idx]
sorted_deltas = deltas[idx]
sorted_cs = [C_TEAL if d > 200 else C_PURPLE if d > 100 else C_BLUE for d in sorted_deltas]

ax2.barh(range(len(sorted_deltas)), sorted_deltas, color=sorted_cs, edgecolor='white', linewidth=0.3)
ax2.set_yticks(range(len(sorted_codes)))
ax2.set_yticklabels(sorted_codes, fontsize=7)
ax2.axvline(x=mean_d, color=C_RED, linestyle='--', alpha=0.5, linewidth=1, label=f'Mean = +{mean_d:.0f}')
ax2.set_xlabel(r'$\Delta$ (CAGate $-$ NOTEARS, edges)')
ax2.set_xlim(right=max(sorted_deltas) * 1.08)  # room for text labels
ax2.set_title(f'(b) Ranked by Delta', fontsize=11, fontweight='bold')

for i, d in enumerate(sorted_deltas):
    xpos = d + 3
    ax2.text(xpos, i, f'+{d:.0f}', va='center', fontsize=6.5, fontweight='bold', color=C_DARK)

ax2.legend(fontsize=7, loc='lower right', framealpha=0.85)
ax2.text(0.02, 1.02, 'b', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='bottom')

plt.tight_layout(pad=1.5)
plt.savefig(os.path.join(FIG_DIR, 'FigS6_RankedByDelta.png'), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'FigS6_RankedByDelta.pdf'), bbox_inches='tight')
plt.close()
print('FigS6 saved')
