"""FigS3: Deep-dive analysis of CAGate performance across 33 TCGA cancer types at d=100.
Data: mega_33_full.json (real experimental values).
Colorblind-friendly: teal/purple/blue palette.
"""
import os
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

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
    rows.append((k, n, cag, nte, delta))

rows.sort(key=lambda x: x[4], reverse=True)
codes = [r[0] for r in rows]
ns = np.array([r[1] for r in rows])
deltas = np.array([r[4] for r in rows])
bases = np.array([r[3] for r in rows])
gates = np.array([r[2] for r in rows])
print(f'FigS3: {len(rows)} cancers, mean delta={np.mean(deltas):.0f}, median={np.median(deltas):.0f}')

# Colorblind-friendly: teal (massive), purple (large), blue (moderate)
C_TEAL   = '#00A087'
C_PURPLE = '#7E57C2'
C_BLUE   = '#3C5488'
C_RED    = '#E64B35'
C_GRAY   = '#95A5A6'
C_DARK   = '#2C3E50'

cs = []
for d in deltas:
    if d > 200: cs.append(C_TEAL)
    elif d > 100: cs.append(C_PURPLE)
    else: cs.append(C_BLUE)

fig, axes = plt.subplots(2, 3, figsize=(16, 10.5))

# ── (a) Delta vs Sample Size ──
ax = axes[0, 0]
ax.scatter(ns, deltas, c=cs, s=70, edgecolors='white', linewidth=0.5, zorder=3, alpha=0.88)
r, p = stats.spearmanr(ns, deltas)
ax.set_xlabel(r'Sample size $n$')
ax.set_ylabel(r'$\Delta$ (CAGate $-$ NOTEARS, edges)')
ax.set_title(r'$\Delta$ vs $n$' + f'\nSpearman $\\rho$ = {r:.3f}, $P$ = {p:.1e}', fontsize=10.5, fontweight='bold')

# Label extreme points: simple offset unless crowded (distance < 15)
label_idx = [i for i in range(len(codes)) if deltas[i] > 300 or deltas[i] < 35]
# Check which points are crowded: ACC(79,320) vs UVM(80,318) → distance 2.2
crowded = set()
for i in range(len(label_idx)):
    for j in range(i+1, len(label_idx)):
        pi, pj = label_idx[i], label_idx[j]
        dist = np.sqrt((ns[pi]-ns[pj])**2 + (deltas[pi]-deltas[pj])**2)
        if dist < 15:
            crowded.add(pi); crowded.add(pj)

for i in label_idx:
    if i in crowded:
        # Arrow from the side for close pairs
        side = 1 if i % 2 == 0 else -1
        ax.annotate(codes[i], xy=(ns[i], deltas[i]), xytext=(ns[i]+side*8, deltas[i]+5*side),
                    fontsize=6, fontweight='bold', color=cs[i],
                    arrowprops=dict(arrowstyle='->', color=cs[i], lw=0.4, alpha=0.6),
                    bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.85))
    else:
        # Simple text next to point
        ax.text(ns[i]+3, deltas[i]+3, codes[i], fontsize=6, fontweight='bold',
                color=cs[i], alpha=0.85)

# ── (b) Delta vs Baseline Edges ──
ax = axes[0, 1]
ax.scatter(bases, deltas, c=cs, s=70, edgecolors='white', linewidth=0.5, zorder=3, alpha=0.88)
r2, p2 = stats.spearmanr(bases, deltas)
ax.set_xlabel('Baseline NOTEARS edges')
ax.set_ylabel(r'$\Delta$ (CAGate $-$ NOTEARS, edges)')
ax.set_title(r'$\Delta$ vs Baseline' + f'\nSpearman $\\rho$ = {r2:.3f}, $P$ = {p2:.1e}', fontsize=10.5, fontweight='bold')
# Label extreme points (plain text, no arrows needed here — x-axis well spread)
label_idx_b = [i for i in range(len(codes)) if bases[i] >= 20 or deltas[i] > 300]
for i in label_idx_b:
    ax.text(bases[i]+0.3, deltas[i]+8, codes[i], fontsize=6, fontweight='bold',
            color=cs[i], alpha=0.85)

# ── (c) Kruskal-Wallis by sample-size tertile ──
ax = axes[0, 2]
tertile_edges = np.percentile(ns, [0, 33.3, 66.7, 100])
t1 = deltas[ns <= tertile_edges[1]]
t2 = deltas[(ns > tertile_edges[1]) & (ns <= tertile_edges[2])]
t3 = deltas[ns > tertile_edges[2]]
groups = {'Small\n(n<=%d)'%tertile_edges[1]: t1,
          'Medium\n(n=%d-%d)'%(int(tertile_edges[1])+1,int(tertile_edges[2])): t2,
          'Large\n(n>%d)'%tertile_edges[2]: t3}
bp = ax.boxplot(groups.values(), patch_artist=True, widths=0.5,
                medianprops=dict(color=C_DARK, linewidth=2))
for patch, color in zip(bp['boxes'], [C_TEAL, C_PURPLE, C_BLUE]):
    patch.set_facecolor(color); patch.set_alpha(0.25)
ax.set_xticklabels(groups.keys(), fontsize=8)
ax.set_ylabel(r'$\Delta$ (edges)')
h, pk = stats.kruskal(t1, t2, t3)
ax.set_title(f'Kruskal-Wallis by $n$ Tertile\n$H$={h:.1f}, $P$={pk:.1e}', fontsize=10.5, fontweight='bold')

# ── (d) Top 10 edge-discovery efficiency ──
ax = axes[1, 0]
eff = gates / ns
top10 = np.argsort(eff)[::-1][:10]
ax.barh(range(10), eff[top10], color=[cs[i] for i in top10], edgecolor='white', linewidth=0.3)
ax.set_yticks(range(10))
ax.set_yticklabels([codes[i] for i in top10], fontsize=7.5)
ax.set_xlabel('Edges per sample')
ax.set_title('Top 10: Discovery Efficiency', fontsize=10.5, fontweight='bold')
for i, idx in enumerate(top10):
    ax.text(eff[idx] + 0.01, i, f'{eff[idx]:.2f}', va='center', fontsize=6.5, fontweight='bold')
ax.invert_yaxis()

# ── (e) Delta histogram ──
ax = axes[1, 1]
ax.hist(deltas, bins=12, color=C_BLUE, edgecolor='white', alpha=0.6, linewidth=0.3)
ax.axvline(np.mean(deltas), color=C_RED, linestyle='--', linewidth=1.5,
           label=f'Mean = {np.mean(deltas):.0f}')
ax.axvline(np.median(deltas), color=C_PURPLE, linestyle=':', linewidth=1.5,
           label=f'Median = {np.median(deltas):.0f}')
skew = stats.skew(deltas)
ax.set_xlabel(r'$\Delta$ (edges)')
ax.set_ylabel('Cancer types')
ax.set_title(r'$\Delta$ Distribution' + f'\nMean={np.mean(deltas):.0f}, Median={np.median(deltas):.0f}, Skew={skew:.2f}',
             fontsize=10.5, fontweight='bold')
ax.legend(fontsize=7)

# ── (f) Small vs Large sample amplification ──
ax = axes[1, 2]
threshold = 323
small_mask = ns < threshold
large_mask = ns >= threshold
small_d = deltas[small_mask]; large_d = deltas[large_mask]
U, pu = stats.mannwhitneyu(small_d, large_d, alternative='greater')

bp2 = ax.boxplot([small_d, large_d], patch_artist=True, widths=0.4,
                 medianprops=dict(color=C_DARK, linewidth=2))
bp2['boxes'][0].set_facecolor(C_TEAL); bp2['boxes'][0].set_alpha(0.30)
bp2['boxes'][1].set_facecolor(C_BLUE); bp2['boxes'][1].set_alpha(0.30)
ax.set_xticklabels([f'Small (n<323)\nMean={np.mean(small_d):.0f}±{np.std(small_d):.0f}',
                    f'Large (n>=323)\nMean={np.mean(large_d):.0f}±{np.std(large_d):.0f}'], fontsize=8)
ax.set_ylabel(r'$\Delta$ (edges)')
ax.set_title(f'Small-Sample Amplification\nMW $U$={U:.0f}, $P$={pu:.4f}', fontsize=10.5, fontweight='bold')

# ── Panel labels ──
for ax_i, label in zip(axes.flat, ['a', 'b', 'c', 'd', 'e', 'f']):
    ax_i.text(-0.02, 1.02, label, transform=ax_i.transAxes, fontsize=12,
              fontweight='bold', va='bottom', ha='left')

plt.tight_layout(pad=2.0)
plt.savefig(os.path.join(FIG_DIR, 'FigS3_DeepDivePanel.png'), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'FigS3_DeepDivePanel.pdf'), bbox_inches='tight')
plt.close()
print('FigS3 saved')
