"""FigS5: KEGG pathway enrichment of CAGate-discovered genes across 10 TCGA cancers.
Data: 05_Replication_Package.zip results/validation/kegg_enrichment.json.
Complements main paper Fig4 (clinical validation) with functional/pathway validation.
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

VAL_DIR = os.path.join(PKG_ROOT, 'results', 'validation')
FIG_DIR = os.path.join(PKG_ROOT, 'figures_supplementary')
os.makedirs(FIG_DIR, exist_ok=True)

with open(os.path.join(VAL_DIR, 'kegg_enrichment.json')) as f:
    kegg = json.load(f)

N_BLUE  = '#3C5488'; N_PURPLE = '#7E57C2'; N_RED = '#E64B35'
N_GRAY  = '#95A5A6'; N_DARK  = '#2C3E50'; N_TEAL = '#00A087'

cancers_sorted = sorted(kegg.keys())
n_pathways = [len(kegg[c]) for c in cancers_sorted]
nonsig = sum(1 for n in n_pathways if n == 0)
print(f'Pathway enrichment: {sum(n_pathways)} total across {len(cancers_sorted)} cancers ({nonsig} with none)')

# Collect all unique pathway names and build presence matrix
all_pathways = set()
for c in cancers_sorted:
    for p in kegg[c]:
        name = p['pathway'].replace('KEGG_', '').replace('_', ' ').title()
        all_pathways.add(name)

# Only show pathways appearing in >= 2 cancers
pathway_counts = {}
for c in cancers_sorted:
    for p in kegg[c]:
        name = p['pathway'].replace('KEGG_', '').replace('_', ' ').title()
        pathway_counts[name] = pathway_counts.get(name, 0) + 1

common = [(name, cnt) for name, cnt in pathway_counts.items() if cnt >= 2]
common.sort(key=lambda x: x[1], reverse=True)
if len(common) > 15:
    common = common[:15]
common_names = [c[0] for c in common]

# Build heatmap matrix: cancers x pathways, value = -log10(p) or 0
heatmap = np.zeros((len(common_names), len(cancers_sorted)))
for j, cancer in enumerate(cancers_sorted):
    for p in kegg[cancer]:
        name = p['pathway'].replace('KEGG_', '').replace('_', ' ').title()
        if name in common_names:
            i = common_names.index(name)
            heatmap[i, j] = -np.log10(max(p['p_value'], 1e-10))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [1, 2.5]})

# ── (a) Bar: enriched pathways per cancer ──
colors_bar = [N_GRAY if n == 0 else N_BLUE for n in n_pathways]
ax1.bar(range(len(cancers_sorted)), n_pathways, color=colors_bar, edgecolor='white', linewidth=0.3)
ax1.set_xticks(range(len(cancers_sorted)))
ax1.set_xticklabels(cancers_sorted, fontsize=7.5, rotation=45, ha='right')
ax1.set_ylabel('Enriched KEGG pathways (p < 0.05)')
ax1.set_title(f'KEGG Enrichment per Cancer\n{sum(n_pathways)} pathways across {len(cancers_sorted)-nonsig}/{len(cancers_sorted)} cancers',
              fontsize=11, fontweight='bold')
for i, n in enumerate(n_pathways):
    if n > 0:
        ax1.text(i, n + 0.2, str(n), ha='center', fontsize=7.5, fontweight='bold', color=N_DARK)
ax1.text(-0.02, 1.02, 'a', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='bottom')

# ── (b) Heatmap: pathways x cancers ──
im = ax2.imshow(heatmap, aspect='auto', cmap='YlOrRd', vmin=0, vmax=np.ceil(np.max(heatmap)))
ax2.set_xticks(range(len(cancers_sorted)))
ax2.set_xticklabels(cancers_sorted, fontsize=7.5, rotation=45, ha='right')
ax2.set_yticks(range(len(common_names)))
ax2.set_yticklabels(common_names, fontsize=7)

# Annotate with gene count and -log10(p)
for i in range(len(common_names)):
    for j in range(len(cancers_sorted)):
        if heatmap[i, j] > 0:
            # Find the original entry to get n_overlap
            name = common_names[i]
            for p in kegg[cancers_sorted[j]]:
                pname = p['pathway'].replace('KEGG_', '').replace('_', ' ').title()
                if pname == name:
                    n_overlap = p['n_overlap']
                    break
            txt = f'{n_overlap}\n({-np.log10(max(p["p_value"], 1e-10)):.1f})'
            ax2.text(j, i, txt, ha='center', va='center', fontsize=5.5,
                     color='white' if heatmap[i, j] > 3 else N_DARK, fontweight='bold')

ax2.set_title(f'Pathway Enrichment Matrix\nTop {len(common_names)} pathways (>=2 cancers), cells: n_genes ( -log10 P )',
              fontsize=11, fontweight='bold')

cbar = fig.colorbar(im, ax=ax2, shrink=0.8, pad=0.02)
cbar.set_label(r'$-\log_{10}(P)$', fontsize=7)
cbar.ax.tick_params(labelsize=6)
ax2.text(-0.02, 1.02, 'b', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'FigS5_DrugTargetBubble.png'), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'FigS5_DrugTargetBubble.pdf'), bbox_inches='tight')
plt.close()
print('FigS5 saved (KEGG pathway enrichment)')
