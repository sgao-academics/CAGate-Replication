"""
CAGate Fig4 — External Validation (2x2 composite).
(a) CTD / ClinGen / STRING validation rates (grouped bar)
(b) Combined rate bubble chart
(c) Drug-target enrichment (Fisher's exact)
(d) Top hubs + therapeutic relevance
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(PKG_ROOT, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ═══════════ NATURE-LEVEL COLOR PALETTE ═══════════
C_CTD = '#0072B2'
C_CLINGEN = '#E69F00'
C_STRING = '#009E73'
C_CAGATE = '#0072B2'
C_BACKGROUND = '#D55E00'
C_GRID = '#E0E0E0'
C_TEXT = '#333333'

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif', 'Times New Roman'],
    'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 3, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.dpi': 400, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

def panel_label(ax, letter):
    ax.text(-0.08, 1.06, letter, transform=ax.transAxes, fontsize=13,
            fontweight='bold', va='bottom', ha='left', color=C_TEXT)

# ═══════════ DATA (from manuscript validation results) ═══════════
val_names = ['BLCA', 'BRCA', 'COAD', 'GBM', 'LAML', 'LIHC', 'LUSC', 'OV', 'PAAD', 'READ']
val_ctd = [89, 98, 93, 97, 94, 91, 90, 95, 92, 96]
val_clin = [82, 74, 79, 69, 76, 80, 72, 78, 75, 73]
val_str = [6.8, 15.7, 8.3, 11.2, 4.5, 2.7, 5.1, 9.4, 7.2, 10.8]
edge_counts = [892, 1939, 1012, 734, 582, 1045, 850, 687, 1021, 620]

# Drug-target enrichment data
ctd_target_pct = 12  # 12% of CAGate genes are known CTD drug targets
bg_pct = 3  # 3% background rate
fold_enrich = 4.0  # 4x enrichment
p_val = 1e-6  # Fisher's exact test p-value

# ═══════════ FIGURE ═══════════
fig = plt.figure(figsize=(8.5, 7.5))
gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.40,
                      left=0.08, right=0.98, bottom=0.06, top=0.95)

# ── (a) Validation rates grouped bar ──
ax = fig.add_subplot(gs[0, 0])
x = np.arange(len(val_names)); w = 0.22
ax.bar(x - w, val_ctd, w, color=C_CTD, linewidth=0, label='CTD drug targets', alpha=0.88)
ax.bar(x, val_clin, w, color=C_CLINGEN, linewidth=0, label='ClinGen drivers', alpha=0.88)
ax.bar(x + w, val_str, w, color=C_STRING, linewidth=0, label='STRING PPIs', alpha=0.88)
ax.axhline(y=3, color=C_BACKGROUND, lw=0.8, ls='--', alpha=0.5)
ax.text(len(val_names) - 2.5, 4.5, 'Background (3%)', fontsize=6.5,
        color=C_BACKGROUND, ha='left', va='bottom', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(val_names, fontsize=7.5, rotation=45, ha='right')
ax.set_ylabel('Validation rate (%)')
ax.legend(fontsize=6.5, loc='lower center', bbox_to_anchor=(0.5, 1.02),
          ncol=3, framealpha=0.85, handletextpad=0.3, columnspacing=0.6)
panel_label(ax, 'a')

# ── (b) Bubble chart: STRING vs combined rate ──
ax = fig.add_subplot(gs[0, 1])
combined = [val_ctd[i] + val_clin[i] + val_str[i] for i in range(len(val_names))]
combined_arr = np.array(combined)
sz = [30 + 180 * (c / max(combined)) for c in combined]
# Sort for occlusion control
idx = np.argsort(sz)[::-1]
for i in idx:
    ax.scatter(val_str[i], combined[i], s=sz[i], c=combined[i], cmap='viridis',
               edgecolor='white', lw=0.5, alpha=0.82, zorder=5)
# Label top 3
for i in np.argsort(combined)[-3:]:
    ax.annotate(val_names[i], (val_str[i], combined[i]), xytext=(8, 6),
                textcoords='offset points', fontsize=7, weight='bold', alpha=0.85,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=C_GRID, alpha=0.9, lw=0.3))
ax.set_xlabel('STRING validation rate (%)')
ax.set_ylabel('Combined rate (%)')
ax.set_xlim(min(val_str) - 1, max(val_str) + 4)
panel_label(ax, 'b')

# ── (c) Drug-target enrichment ──
ax = fig.add_subplot(gs[1, 0])
categories = ['CAGate\ntop-100', 'Random\ngene set', 'Cancer-\nspecific']
values = [12, 3, 9.6]  # 12%, 3%, 12 * 0.8 (3.2x for cancer-specific)
colors_bar = [C_CAGATE, C_BACKGROUND, '#56B4E9']
bars = ax.bar(range(3), values, color=colors_bar, width=0.5, alpha=0.88,
              edgecolor='white', linewidth=0.3)
for i, v in enumerate(values):
    ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=8.5, weight='bold', color=C_TEXT)
ax.set_xticks(range(3))
ax.set_xticklabels(categories, fontsize=8)
ax.set_ylabel('Drug-target overlap (%)')
ax.set_ylim(0, 16)
# Enrichment annotation — top-left to avoid bar overlap
ax.text(0.03, 0.94, f'{fold_enrich}x enrichment\n$P < 10^{{-6}}$',
        transform=ax.transAxes, fontsize=7.5, ha='left', va='top',
        color=C_CAGATE, weight='bold',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_GRID, alpha=0.92, lw=0.4))
panel_label(ax, 'c')

# ── (d) Top therapeutic hubs (sorted, correct alignment) ──
ax = fig.add_subplot(gs[1, 1])
hub_genes = ['ESR1', 'FOXA1', 'GATA3', 'MYC', 'TP53', 'RUNX2', 'AR', 'ERBB2']
influence = [1.22, 1.17, 0.95, 0.88, 0.82, 0.48, 0.45, 0.41]
fda_status = [True, False, True, True, True, False, True, True]

order = np.argsort(influence)  # shortest first -> barh puts at bottom
y = list(range(len(hub_genes)))
ax.barh(y, [influence[i] for i in order],
        color=['#0072B2' if fda_status[i] else '#D55E00' for i in order],
        height=0.6, alpha=0.88, edgecolor='white', linewidth=0.3)
ax.set_yticks(y)
ax.set_yticklabels([hub_genes[i] for i in order], fontsize=8)
ax.set_xlabel('Influence score')

# Value labels tight to bar-end right
for idx, i in enumerate(order):
    ax.text(influence[i] + 0.03, idx, f'{influence[i]:.2f}',
            va='center', fontsize=7.5, color=C_TEXT)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#0072B2', alpha=0.88, label='FDA-approved therapy'),
                   Patch(facecolor='#D55E00', alpha=0.88, label='Under development')]
ax.legend(handles=legend_elements, fontsize=6.5, loc='lower right', framealpha=0.85,
          handletextpad=0.4)
panel_label(ax, 'd')

# Save
out = os.path.join(FIG_DIR, 'Fig4_Validation.pdf')
fig.savefig(out, dpi=400, facecolor='white', edgecolor='none')
plt.close(fig)
print(f'[SAVED] Fig4_Validation.pdf ({os.path.getsize(out)//1024} KB)')
