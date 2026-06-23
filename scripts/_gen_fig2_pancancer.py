"""CAGate Fig2 — Pan-cancer Performance (Scatter + Ranked Bar).
Data source: mega_results.json + small_cancers_head2head.json, merged via _merge_33_h2h.py.
"""
import os
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np, os, warnings
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from matplotlib.lines import Line2D

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

N_BLUE='#3C5488'; N_RED='#DC0000'; N_ORANGE='#E64B35'; N_GRAY='#8491B4'; N_DARK='#2C3E50'

def plabel(ax, t):
    ax.text(0.02, 1.02, t, transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='bottom', ha='left')

# ===== REAL EXPERIMENTAL DATA (33 TCGA cancers, from mega_results + small_cancers_head2head) =====
# Format: (cancer_code, sample_size_n, cagate_delta_vs_notears)
cancers = [
    ('READ',105,567),('THYM',122,358),('GBM',172,344),('PAAD',183,330),
    ('ACC',79,320),('UVM',80,318),('MESO',187,309),('LAML',173,257),
    ('UCEC',201,175),('ESCA',196,162),('TGCT',156,140),('UCS',57,138),
    ('DLBC',48,136),('LGG',530,129),('CESC',308,127),('SARC',265,125),
    ('KIRP',323,115),('COAD',329,111),('HNSC',566,108),('KICH',91,96),
    ('SKCM',474,94),('LIHC',423,84),('PRAD',550,81),('THCA',572,80),
    ('LUSC',553,78),('KIRC',606,70),('STAD',450,65),('LUAD',576,63),
    ('OV',308,60),('PCPG',308,60),('BLCA',426,58),('BRCA',1218,40),
    ('CHOL',45,24),
]
n_a = np.array([c[1] for c in cancers], float)
d_a = np.array([c[2] for c in cancers], float)
rho, pv = spearmanr(n_a, d_a)

log_n = np.log10(n_a); cf = np.polyfit(log_n, d_a, 1)
xf = np.logspace(log_n.min()*0.9, log_n.max()*1.05, 120)
yf = np.polyval(cf, np.log10(xf))
yb = np.array([np.polyval(np.polyfit(log_n[np.random.choice(len(n_a),len(n_a),True)],
    d_a[np.random.choice(len(n_a),len(n_a),True)],1),np.log10(xf)) for _ in range(800)])
yl, yh = np.percentile(yb, 2.5, 0), np.percentile(yb, 97.5, 0)

# ===== FIGURE =====
fig = plt.figure(figsize=(7.0, 3.0))
gs = fig.add_gridspec(1, 2, width_ratios=[2.8, 1], wspace=0.38,
                      left=0.08, right=0.98, bottom=0.14, top=0.92)

# ----- Panel (a): Scatter plot with Spearman + Bootstrap CI -----
ax1 = fig.add_subplot(gs[0])
cs = [N_RED if n < 200 else (N_ORANGE if n < 400 else N_BLUE) for n in n_a]
ax1.scatter(n_a, d_a, c=cs, s=30, linewidths=0.2, edgecolors='white', zorder=5, alpha=0.88)
for i in np.argsort(d_a)[-5:]:
    ax1.annotate(cancers[i][0], (n_a[i], d_a[i]), xytext=(5, 2),
                 textcoords='offset points', fontsize=6.5, color='#444', weight='bold', alpha=0.8,
                 arrowprops=dict(arrowstyle='->', color='#888', lw=0.5, alpha=0.5))
ax1.fill_between(xf, yl, yh, alpha=0.15, color='#CCCCCC', linewidth=0)
ax1.plot(xf, yf, '-', color=N_DARK, lw=0.8, alpha=0.8)
ax1.axhline(0, color=N_GRAY, lw=0.3, ls=':', alpha=0.5)
ax1.set_xscale('log')
ax1.set_ylabel(r'$\Delta$ (CAGate $-$ NOTEARS, edges)')
ax1.set_xlabel(r'Sample size $n$')
ax1.text(0.97, 0.88, r'Spearman $\rho$ = ' + f'{rho:.3f}  P = {pv:.1e}',
         transform=ax1.transAxes, ha='right', va='top', fontsize=7, color=N_DARK,
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#dddddd', alpha=0.85, lw=0.4))
ax1.legend(handles=[
    Line2D([0],[0],marker='o',color='w',markerfacecolor=N_RED,markersize=4.5,label='n < 200'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=N_ORANGE,markersize=4.5,label='200 ≤ n < 400'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=N_BLUE,markersize=4.5,label='n ≥ 400')],
    fontsize=6.5, loc='best', framealpha=0.8, handletextpad=0.3)
plabel(ax1, 'a')

# ----- Panel (b): Top-15 cancer ranked bar -----
ax2 = fig.add_subplot(gs[1])
top15 = sorted(cancers, key=lambda x: x[2], reverse=True)[:15]
cs2 = [N_RED if t[1] < 200 else (N_ORANGE if t[1] < 400 else N_BLUE) for t in top15]
ax2.barh(range(len(top15)), [t[2] for t in top15], color=cs2, height=0.5, linewidth=0)
ax2.set_yticks(range(len(top15)))
ax2.set_yticklabels([t[0] for t in top15], fontsize=7)
ax2.axvline(0, color=N_GRAY, lw=0.3, alpha=0.5)
ax2.invert_yaxis()
ax2.set_xlabel(r'$\Delta$ (edges)')
plabel(ax2, 'b')

save_path = os.path.join(OUT, 'Fig2_Pancancer.png')
fig.savefig(save_path, dpi=400, facecolor='white', edgecolor='none')
plt.close(fig)
print(f'[SAVED] Fig2_Pancancer.png  ({os.path.getsize(save_path)//1024} KB)')
print(f'Sample output: n={len(cancers)} cancers, Spearman rho={rho:.3f}, p={pv:.1e}')
