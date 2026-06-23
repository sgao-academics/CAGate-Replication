"""
CAGate Mechanism Figure — Publication-quality redraw.
Figure 1: Cluster-Aware Gating mechanism diagram.
Clean, modern, professional — no gridlines, warm/cool contrast, clear visual story.
"""
import os
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# ── Publication-grade font: LaTeX-consistent serif + Computer Modern math ──
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.unicode_minus'] = False

# ── Color system ──────────────────────────────────────
C_BAD    = '#E74C3C'   # NOTEARS noise (warm red-orange)
C_GOOD   = '#2980B9'   # CAGate signal (cool blue)
C_GATE   = '#27AE60'   # Gate weights (green)
C_DARK   = '#2C3E50'   # Text
C_GRAY   = '#95A5A6'   # Subtle elements
C_WARN   = '#E67E22'   # Warning/emphasis
C_PURPLE = '#8E44AD'   # Cluster C (avoids orange-on-orange clash with gate badge)

FIGSIZE = (16, 7)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE, facecolor='white')
for ax in (ax1, ax2):
    ax.set_facecolor('white')
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 10)    # extra bottom margin so text boxes are not clipped
    ax.axis('off')

# ═══════════════════════════════════════════════════
# PANEL (a): NOTEARS — pooled samples, diluted signal
# ═══════════════════════════════════════════════════

ax1.text(5, 9.6, 'NOTEARS: Pooled Optimization', ha='center', va='top',
         fontsize=14, color=C_BAD, fontweight='bold')

# ── Three sample clusters (scattered dots) ──
clusters_a = [
    {'center': (3, 6.5), 'color': '#3498DB', 'n': 80, 'spread': 0.6},
    {'center': (7, 6.5), 'color': '#E74C3C', 'n': 80, 'spread': 1.2},
    {'center': (5, 3.5), 'color': C_PURPLE, 'n': 80, 'spread': 1.5},
]

for cl in clusters_a:
    rng = np.random.RandomState(42 if cl['center'][0] == 3 else (7 if cl['center'][0] == 7 else 13))
    xs = cl['center'][0] + rng.randn(cl['n']) * cl['spread']
    ys = cl['center'][1] + rng.randn(cl['n']) * cl['spread']
    # Clip to prevent dots escaping axis bounds
    xs = np.clip(xs, 0.5, 9.5)
    ys = np.clip(ys, 0.5, 9.5)
    ax1.scatter(xs, ys, c=cl['color'], s=25, alpha=0.6, edgecolors='white', linewidth=0.3, zorder=3)

# ── Big dashed box: "Pool all together" ──
pool_box = FancyBboxPatch((0.3, 1.2), 9.4, 7.6,
                          boxstyle="round,pad=0.1,rounding_size=0.5",
                          facecolor='none', edgecolor=C_BAD, linewidth=2.5,
                          linestyle='--', zorder=2, alpha=0.8)
ax1.add_patch(pool_box)
ax1.text(5, 8.5, 'All samples pooled together', ha='center', va='center',
         fontsize=11, color=C_BAD, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C_BAD, alpha=0.9),
         zorder=5)

# ── Gradient dilution arrow ──
arr = FancyArrowPatch((5, 8.8), (5, 1.5),
                      arrowstyle='->', mutation_scale=30,
                      color=C_BAD, linewidth=3, alpha=0.5, zorder=1,
                      connectionstyle="arc3,rad=0")
ax1.add_patch(arr)

# Bottom: diluted result
ax1.text(5, -0.2, 'Gradient signal diluted into\nheterogeneous noise', ha='center', va='top',
         fontsize=12, color=C_BAD, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#FDEDEC', edgecolor=C_BAD, alpha=0.8),
         zorder=5)

# ═══════════════════════════════════════════════════
# PANEL (b): CAGate — cluster-aware gating
# ═══════════════════════════════════════════════════

ax2.text(5, 9.6, 'CAGate: Cluster-Aware Gating', ha='center', va='top',
         fontsize=14, color=C_GOOD, fontweight='bold')

# ── Three separate cluster boxes ──
clusters_b = [
    {'center': (1.8, 5.0), 'color': '#3498DB', 'label': 'Cluster A', 'gate': 0.90,
     'desc': 'Homogeneous\nStrong signal', 'w': 2.4, 'h': 4.0},
    {'center': (5.0, 5.0), 'color': '#E74C3C', 'label': 'Cluster B', 'gate': 0.15,
     'desc': 'Heterogeneous\nWeak signal', 'w': 2.4, 'h': 4.0},
    {'center': (8.2, 5.0), 'color': C_PURPLE, 'label': 'Cluster C', 'gate': 0.15,
     'desc': 'Mixed\nNoise-dominant', 'w': 2.4, 'h': 4.0},
]

for cl in clusters_b:
    cx, cy = cl['center']
    w, h = cl['w'], cl['h']
    # Cluster box
    box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle="round,pad=0.08,rounding_size=0.25",
                          facecolor='white', edgecolor=cl['color'], linewidth=2.0, zorder=2)
    ax2.add_patch(box)
    # Dots inside cluster
    rng = np.random.RandomState(42 if 'A' in cl['label'] else (7 if 'B' in cl['label'] else 13))
    spread = 0.25 if 'A' in cl['label'] else (0.5 if 'B' in cl['label'] else 0.7)
    xs = cx + rng.randn(60) * spread
    ys = cy + rng.randn(60) * spread
    xs = np.clip(xs, cx - w/2 + 0.2, cx + w/2 - 0.2)
    ys = np.clip(ys, cy - h/2 + 0.2, cy + h/2 - 0.2)
    ax2.scatter(xs, ys, c=cl['color'], s=18, alpha=0.5, edgecolors='none', zorder=3)

    # Cluster label (all serif, consistent with global font)
    ax2.text(cx, cy + h/2 + 0.25, cl['label'], ha='center', va='bottom',
             fontsize=10, color=cl['color'], fontweight='bold', zorder=4)

    # Gate weight badge
    gate_color = C_GATE if cl['gate'] > 0.5 else (C_WARN if cl['gate'] > 0.1 else C_GRAY)
    gate_alpha = 1.0 if cl['gate'] > 0.5 else (0.7 if cl['gate'] > 0.1 else 0.4)
    gx, gy = cx, cy - h/2 - 0.6
    gate_label = f"γ = {cl['gate']:.2f}"
    ax2.text(gx, gy, gate_label, ha='center', va='center',
             fontsize=13, color='white', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.25', facecolor=gate_color, alpha=gate_alpha),
             zorder=5)

    # Description below gate (widened gap: gy - 0.8 instead of 0.55)
    ax2.text(gx, gy - 0.8, cl['desc'], ha='center', va='top',
             fontsize=8, color=C_GRAY, zorder=4)

# ── Arrows from clusters to output (stopped above bottom text) ──
ARROW_Y = 1.5   # stop arrows well above the bottom result text box
ax2.annotate('', xy=(1.8, ARROW_Y), xytext=(1.8, 2.3),
             arrowprops=dict(arrowstyle='->', mutation_scale=25,
                           color=C_GOOD, lw=4, alpha=0.9),
             zorder=4)
ax2.annotate('', xy=(5.0, ARROW_Y), xytext=(5.0, 2.3),
             arrowprops=dict(arrowstyle='->', mutation_scale=15,
                           color=C_GRAY, lw=1.5, alpha=0.4),
             zorder=4)
ax2.annotate('', xy=(8.2, ARROW_Y), xytext=(8.2, 2.3),
             arrowprops=dict(arrowstyle='->', mutation_scale=15,
                           color=C_GRAY, lw=1.0, alpha=0.3),
             zorder=4)

# ── MAD-aware gate formula ──
ax2.text(5, 8.8, r'MAD-aware Gate Weights:  $\gamma_k = 1 - \exp(-\alpha \cdot \overline{\mathrm{MAD}}_k)$',
         ha='center', va='center',
         fontsize=12, color=C_GOOD, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#EBF5FB', edgecolor=C_GOOD, alpha=0.9),
         zorder=5)

# ── Bottom: result (moved up to avoid clipping) ──
ax2.text(5, 0.8, 'Focused optimization on homogeneous clusters\n→ Clean gradient → Accurate DAG recovery',
         ha='center', va='top',
         fontsize=12, color=C_GOOD, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#EBF5FB', edgecolor=C_GOOD, alpha=0.9, linewidth=2),
         zorder=5)

# ── Panel labels ──
for ax, label in [(ax1, '(a)'), (ax2, '(b)')]:
    ax.text(0.15, 9.85, label, ha='left', va='top',
            fontsize=16, color=C_DARK, fontweight='bold')

# ── Figure caption (brief; full caption in LaTeX) ──
fig.text(0.5, 0.02,
         'Figure 1. CAGate mechanism: (a) NOTEARS pools all samples, diluting gradient into noise; '
         '(b) CAGate partitions samples and applies MAD-aware gating.',
         ha='center', va='top', fontsize=9, color=C_DARK,
         fontstyle='italic', wrap=True)

plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.12, wspace=0.08)

# ── Save ──────────────────────────────────────────────
OUT = os.path.join(PKG_ROOT, 'figures')
png_path = os.path.join(OUT, 'Fig1_Mechanism.png')
pdf_path = os.path.join(OUT, 'Fig1_Mechanism.pdf')
fig.savefig(png_path, dpi=250, bbox_inches='tight', facecolor='white', edgecolor='none')
fig.savefig(pdf_path, dpi=250, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
png_size = os.path.getsize(png_path)
pdf_size = os.path.getsize(pdf_path)
print(f"[SAVED] {png_path}  ({png_size:,} bytes = {png_size/1024:.1f} KB)")
print(f"[SAVED] {pdf_path}  ({pdf_size:,} bytes = {pdf_size/1024:.1f} KB)")
print("\nDone. Open the PNG to review.")
