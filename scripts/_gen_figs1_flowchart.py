"""FigS1: CAGate training flowchart. Conceptual diagram of the algorithm.
Shows iterative loop: Input -> PCA Residuals -> K-means Clustering -> MAD-aware Gate
-> Gate-Weighted Loss -> NOTEARS Optimization -> Convergence Check -> Output W*.
"""
import os
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.unicode_minus'] = False

FIG_DIR = os.path.join(PKG_ROOT, 'figures_supplementary')
os.makedirs(FIG_DIR, exist_ok=True)

N_BLUE, N_RED, C_TEAL, N_PURPLE = '#3C5488', '#E64B35', '#00A087', '#7E57C2'
N_ORANGE, N_GRAY, N_DARK = '#F39C12', '#95A5A6', '#2C3E50'
BG = '#F8F9FA'

fig, ax = plt.subplots(figsize=(10, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')
ax.set_facecolor(BG)

def draw_box(ax, x, y, w, h, text, color=N_BLUE, text_color='white', fontsize=9, fontweight='bold'):
    """Draw a rounded box with text."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.1", facecolor=color,
                          edgecolor='white', linewidth=1.5, alpha=0.92, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight=fontweight, zorder=4)

def draw_arrow(ax, x1, y1, x2, y2, color=N_GRAY, lw=1.2, label=None):
    """Draw arrow from (x1,y1) to (x2,y2)."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                               connectionstyle='arc3,rad=0', zorder=2))
    if label is not None:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, fontsize=7, color=N_DARK, ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8), zorder=5)

# ── Box positions: vertical flow, center at x=5 ──
# Step 1: Input
draw_box(ax, 5, 13.2, 4.5, 0.8, r'Input: $\mathbf{X} \in \mathbb{R}^{n \times d}$' + '\n(TCGA expression, $d$ top-variance genes)',
         N_DARK, 'white', 9)
draw_arrow(ax, 5, 12.7, 5, 12.1, N_GRAY)

# Step 2: PCA Residuals
draw_box(ax, 5, 11.6, 4.5, 0.8, 'PCA-Residual Transformation\n' + r'$\mathbf{R} = \mathbf{X} - \mathbf{X}_{\mathrm{PCA}(q=5)}$',
         N_BLUE, 'white', 9)
draw_arrow(ax, 5, 11.1, 5, 10.4, N_GRAY)

# Step 3: K-means Clustering
draw_box(ax, 5, 9.8, 4.5, 0.9, r'$K$-means Clustering on $\mathbf{R}$'+'\n' + r'$K=2,\ldots,K_{\max}$ selected by silhouette',
         N_PURPLE, 'white', 9)
draw_arrow(ax, 5, 9.25, 5, 8.7, N_GRAY, label=r'$\{C_1,\ldots,C_K\}$')

# Step 4: MAD-aware Gate
draw_box(ax, 5, 8.1, 4.8, 1.0, r'MAD-Aware Gate Weights'+'\n' + r'$\overline{\mathrm{MAD}}_k = \mathrm{median}(|\mathbf{R}_{C_k} - \tilde{\mathbf{R}}_{C_k}|)$' + '\n' + r'$\gamma_k = 1 - \exp(-\alpha \cdot \overline{\mathrm{MAD}}_k)$',
         C_TEAL, 'white', 8)
draw_arrow(ax, 5, 7.5, 5, 6.9, N_GRAY, label=r'$\gamma_1,\ldots,\gamma_K$')

# Step 5: Gate-Weighted Loss
draw_box(ax, 5, 6.3, 4.5, 1.0, r'Gate-Weighted Structural Loss'+'\n' + r'$\mathcal{L}_{\gamma}(W) = \sum_{k=1}^{K} \gamma_k \sum_{i \in C_k} \|\mathbf{x}_i - \mathbf{x}_i W\|^2$',
         N_ORANGE, 'white', 8)
draw_arrow(ax, 5, 5.7, 5, 5.1, N_GRAY)

# Step 6: NOTEARS Optimization
draw_box(ax, 5, 4.5, 4.5, 1.0, r'DAG-Constrained Optimization (NOTEARS)'+'\n' + r'$\min_W \mathcal{L}_{\gamma}(W) + \rho \cdot h(W) + \frac{\mu}{2}h(W)^2$'+'\n'+r'$h(W) = \mathrm{tr}(e^{W \odot W}) - d$',
         N_RED, 'white', 8)
draw_arrow(ax, 5, 3.9, 5, 3.3, N_GRAY)

# Step 7: Convergence check (diamond)
diamond = mpatches.Polygon([
    (5, 3.05), (5.9, 2.55), (5, 2.05), (4.1, 2.55)
], facecolor=N_BLUE, edgecolor='white', linewidth=1.5, alpha=0.92, zorder=3)
ax.add_patch(diamond)
ax.text(5, 2.55, r'$h(W) \leq 10^{-8}$?' + '\nor ' + r'$\rho > 10^{16}$?',
        ha='center', va='center', fontsize=8, color='white', fontweight='bold', zorder=4)

# "No" return path (left side)
ax.annotate('', xy=(4.1, 2.55), xytext=(1.5, 2.55),
            arrowprops=dict(arrowstyle='->', color=N_GRAY, lw=1.2, connectionstyle='arc3,rad=0.3'))
ax.annotate('', xy=(1.5, 10.15), xytext=(1.5, 2.55),
            arrowprops=dict(arrowstyle='->', color=N_GRAY, lw=1.2))
ax.annotate('', xy=(2.75, 10.15), xytext=(1.5, 10.15),
            arrowprops=dict(arrowstyle='->', color=N_GRAY, lw=1.2))
ax.text(1.0, 6.35, 'NO\nUpdate\nClusters', ha='center', va='center', fontsize=7.5,
        color=N_DARK, fontweight='bold', rotation=90,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=N_GRAY, alpha=0.9), zorder=5)

# "Yes" path to output
ax.annotate('', xy=(5, 2.05), xytext=(5, 1.4),
            arrowprops=dict(arrowstyle='->', color=C_TEAL, lw=1.5))
ax.text(5.5, 1.7, 'YES', fontsize=7.5, color=C_TEAL, fontweight='bold')

# Step 8: Output
draw_box(ax, 5, 0.9, 4.5, 0.8, r'Output: $\mathbf{W}^*$ (Causal Graph)' + '\n' + r'Cluster assignments $\{C_1,\ldots,C_K\}$',
         C_TEAL, 'white', 9)

# ── Legend ──
legend_items = [
    mpatches.Patch(color=N_DARK, alpha=0.92, label='Input / Output'),
    mpatches.Patch(color=N_BLUE, alpha=0.92, label='Preprocessing'),
    mpatches.Patch(color=N_PURPLE, alpha=0.92, label='Clustering'),
    mpatches.Patch(color=C_TEAL, alpha=0.92, label='Gate Weight'),
    mpatches.Patch(color=N_ORANGE, alpha=0.92, label='Loss Function'),
    mpatches.Patch(color=N_RED, alpha=0.92, label='Optimization'),
]
ax.legend(handles=legend_items, loc='lower right', fontsize=7, framealpha=0.9, ncol=3,
          bbox_to_anchor=(0.95, -0.05))

# ── Title ──
ax.text(5, 14.5, 'Figure S1: CAGate Training Flowchart',
        ha='center', fontsize=14, fontweight='bold', color=N_DARK)
ax.text(5, 14.1, 'MAD = median absolute deviation; PCA = principal component analysis; DAG = directed acyclic graph',
        ha='center', fontsize=7.5, color=N_GRAY, fontstyle='italic')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'FigS1_TrainingFlowchart.png'), dpi=200, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
plt.savefig(os.path.join(FIG_DIR, 'FigS1_TrainingFlowchart.pdf'), bbox_inches='tight',
            facecolor=BG, edgecolor='none')
plt.close()
print('FigS1 saved')
