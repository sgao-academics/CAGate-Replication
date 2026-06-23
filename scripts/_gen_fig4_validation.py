"""CAGate Fig4 — External Validation (Bar Chart + Bubble Chart).
Data source: replication package validation JSONs (drug_targets, cancer_drivers, string).
"""
import os
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np, os, warnings, json
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(PKG_ROOT, 'figures')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family':'serif','font.serif':['DejaVu Serif','Times New Roman'],
    'font.size':7,'axes.labelsize':8,'axes.titlesize':7.5,
    'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':6.5,
    'axes.linewidth':0.5,'xtick.major.width':0.4,'ytick.major.width':0.4,
    'xtick.major.size':2,'ytick.major.size':2,
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.labelpad':2,'figure.facecolor':'white','axes.facecolor':'white',
    'savefig.dpi':400,'savefig.bbox':'tight','savefig.pad_inches':0.03,
})

N_BLUE='#3C5488'; N_ORANGE='#E64B35'; N_PURPLE='#7E57C2'; N_GRAY='#8491B4'; N_DARK='#2C3E50'

def plabel(ax, t):
    ax.text(0.02, 1.02, t, transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='bottom', ha='left')

# ===== LOAD REAL VALIDATION DATA =====
VAL_DIR = os.path.join(PKG_ROOT, 'results', 'validation')
with open(os.path.join(VAL_DIR, 'drug_targets.json')) as f: ctd_data = json.load(f)
with open(os.path.join(VAL_DIR, 'cancer_drivers.json')) as f: clin_data = json.load(f)
with open(os.path.join(VAL_DIR, 'string.json')) as f: str_data = json.load(f)
with open(os.path.join(VAL_DIR, 'cancer_edge_counts.json')) as f: edge_data = json.load(f)

# Common cancer set (10 cancers)
val_names = sorted(ctd_data.keys())  # BLCA, BRCA, GBM, LAML, LIHC, LUSC, OV, PAAD, READ, THYM

val_ctd  = [ctd_data[c]['pct']  for c in val_names]
val_clin = [clin_data[c]['clingen_pct'] for c in val_names]
val_str  = [str_data[c]['pct']  for c in val_names]

# ===== FIGURE =====
fig = plt.figure(figsize=(7.0, 3.0))
gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.38,
                      left=0.08, right=0.98, bottom=0.14, top=0.92)

# ----- Panel (a): Grouped bar chart -----
ax1 = fig.add_subplot(gs[0])
x = np.arange(len(val_names)); w = 0.2
ax1.bar(x-w, val_ctd,  w, color=N_BLUE,   linewidth=0, label='CTD',    zorder=5)
ax1.bar(x,   val_clin, w, color=N_ORANGE, linewidth=0, label='ClinGen', zorder=5)
ax1.bar(x+w, val_str,  w, color=N_PURPLE,  linewidth=0, label='STRING', zorder=5)
# Background rate at 3%
ax1.axhline(y=3, color=N_GRAY, lw=0.5, ls='--', alpha=0.4)
ax1.text(len(val_names), 3.5, 'Background (3%)', fontsize=6,
         color=N_GRAY, ha='left', va='bottom', alpha=0.8,
         bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9))
ax1.set_xticks(x)
ax1.set_xticklabels(val_names, fontsize=7, rotation=45, ha='right')
ax1.set_ylabel(r'Validation rate (\%)')
ax1.legend(fontsize=6.5, loc='lower center', bbox_to_anchor=(0.5, 1.02),
           ncol=3, framealpha=0.8, handletextpad=0.3)
plabel(ax1, 'a')

# ----- Panel (b): Bubble chart -----
ax2 = fig.add_subplot(gs[1])
combined = [val_ctd[i] + val_clin[i] + val_str[i] for i in range(len(val_names))]
sz = [max(25, 30 + 150*(c/max(combined))) for c in combined]
# Sort by bubble size descending: small bubbles paint on top, preventing occlusion
sorted_idx = np.argsort(sz)[::-1]
val_str_sorted = [val_str[i] for i in sorted_idx]
combined_sorted = [combined[i] for i in sorted_idx]
sz_sorted = [sz[i] for i in sorted_idx]

sc = ax2.scatter(val_str_sorted, combined_sorted, s=sz_sorted, c=combined_sorted, cmap='viridis',
                 edgecolors='white', lw=0.5, alpha=0.75, zorder=5)
ax2.set_xlim(min(val_str)-1, max(val_str)+5)
ax2.set_ylim(min(combined)-5, max(combined)+5)
cbar = fig.colorbar(sc, ax=ax2, shrink=0.8, pad=0.12)
cbar.set_label(r'Combined rate (\%)', fontsize=6)
cbar.ax.tick_params(labelsize=5)
for i in np.argsort(combined)[-3:]:
    ax2.annotate(val_names[i], (val_str[i], combined[i]), xytext=(10, 8),
                 textcoords='offset points', fontsize=6.5, weight='bold', alpha=0.85,
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9))
ax2.set_xlabel(r'STRING validation rate (\%)')
ax2.set_ylabel(r'Combined rate (\%)')
plabel(ax2, 'b')

save_path = os.path.join(OUT, 'Fig4_Validation.png')
fig.savefig(save_path, dpi=400, facecolor='white', edgecolor='none')
plt.close(fig)
print(f'[SAVED] Fig4_Validation.png  ({os.path.getsize(save_path)//1024} KB)')

# Print data for LaTeX caption update
print(f'\nValidation cancers: {val_names}')
print(f'CTD range: {min(val_ctd):.0f}-{max(val_ctd):.0f}%')
print(f'ClinGen range: {min(val_clin):.0f}-{max(val_clin):.0f}%')
print(f'STRING range: {min(val_str):.0f}-{max(val_str):.0f}%')
print(f'Combined top: {max(combined):.0f}% ({val_names[np.argmax(combined)]})')
print(f'Edge counts (a area): min={min(edge_data.values())}, max={max(edge_data.values())}')
