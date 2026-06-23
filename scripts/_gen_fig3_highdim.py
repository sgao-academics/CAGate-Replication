"""CAGate Fig3 — High-Dimensional Performance (Edges vs d + Delta vs n)."""
import os
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np, os, warnings
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(PKG_ROOT, 'figures')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family':'serif','font.serif':['DejaVu Serif','Times New Roman'],
    'font.size':7,'axes.labelsize':8,'axes.titlesize':7.5,
    'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6,
    'axes.linewidth':0.5,'xtick.major.width':0.4,'ytick.major.width':0.4,
    'xtick.major.size':2,'ytick.major.size':2,
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.labelpad':2,'figure.facecolor':'white','axes.facecolor':'white',
    'savefig.dpi':400,'savefig.bbox':'tight','savefig.pad_inches':0.03,
})

N_RED='#DC0000'; N_BLUE='#3C5488'; N_GRAY='#8491B4'; N_DARK='#2C3E50'

def plabel(ax, t):
    ax.text(0.02, 1.02, t, transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='bottom', ha='left')

# ===== DATA =====
synth_d = [10, 20, 30, 50, 75, 100, 150, 200]
nte_e = [28, 55, 82, 140, 215, 308, 12, 0]
nte_s = [3, 5, 7, 11, 17, 22, 2, 0]
cag_e = [27, 54, 80, 136, 218, 312, 218, 552]
cag_s = [3, 5, 6, 11, 16, 21, 18, 34]

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
from scipy.stats import spearmanr
rho, pv = spearmanr(n_a, d_a)

log_n = np.log10(n_a); cf = np.polyfit(log_n, d_a, 1)
xf = np.logspace(log_n.min()*0.9, log_n.max()*1.05, 120)
yf = np.polyval(cf, np.log10(xf))
yb = np.array([np.polyval(np.polyfit(log_n[np.random.choice(len(n_a),len(n_a),True)],
    d_a[np.random.choice(len(n_a),len(n_a),True)],1),np.log10(xf)) for _ in range(800)])
yl, yh = np.percentile(yb, 2.5, 0), np.percentile(yb, 97.5, 0)

# ===== FIGURE =====
fig = plt.figure(figsize=(7.0, 3.0))
gs = fig.add_gridspec(1, 2, wspace=0.38, left=0.08, right=0.98, bottom=0.14, top=0.92)

ax1 = fig.add_subplot(gs[0])
ax1.axvspan(150, 205, alpha=0.15, color=N_RED, linewidth=0, zorder=0)
ax1.annotate('Failure zone', xy=(175, 420), xytext=(130, 420),
             arrowprops=dict(arrowstyle='->', color=N_RED, lw=0.8),
             fontsize=6, color=N_RED, weight='bold', ha='center', alpha=0.8,
             bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))
ax1.errorbar(synth_d, cag_e, yerr=cag_s, fmt='o-', color=N_RED, lw=1.8, markersize=4.5,
             capsize=0, elinewidth=0.8, zorder=5, label='CAGate',
             markeredgecolor='white', markeredgewidth=0.2)
ax1.errorbar(synth_d, nte_e, yerr=nte_s, fmt='s--', color=N_GRAY, lw=1.3, markersize=4,
             capsize=0, elinewidth=0.6, zorder=4, label='NOTEARS',
             markeredgecolor='white', markeredgewidth=0.2)
ax1.annotate('552', xy=(200, cag_e[-1]), xytext=(180, cag_e[-1]+18),
             fontsize=6.5, weight='bold', color=N_RED, ha='center',
             arrowprops=dict(arrowstyle='->', color=N_RED, lw=0.6, shrinkA=0, shrinkB=1))
ax1.set_xlabel(r'Dimensionality $d$')
ax1.set_ylabel('Edges recovered')
ax1.set_ylim(bottom=-10, top=620)
ax1.legend(fontsize=6.5, loc='upper left', framealpha=0.8, handletextpad=0.3)
plabel(ax1, 'a')

ax2 = fig.add_subplot(gs[1])
cs3 = [N_RED if n < 323 else N_BLUE for n in n_a]
ax2.scatter(n_a, d_a, c=cs3, s=26, linewidths=0.2, edgecolors='white', zorder=5, alpha=0.88)
ax2.fill_between(xf, yl, yh, alpha=0.15, color='#CCCCCC', linewidth=0)
ax2.plot(xf, yf, '-', color=N_DARK, lw=0.8, alpha=0.8)
ax2.axhline(0, color=N_GRAY, lw=0.3, ls=':', alpha=0.5)
sm = np.mean(d_a[n_a < 323]); lg = np.mean(d_a[n_a >= 323])
ax2.text(0.03, 0.90, f'n<323: {sm:.0f}±{np.std(d_a[n_a<323]):.0f}\nn≥323: {lg:.0f}±{np.std(d_a[n_a>=323]):.0f}',
         transform=ax2.transAxes, ha='left', va='top', fontsize=6.5, color=N_DARK,
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#dddddd', alpha=0.85, lw=0.4))
ax2.text(0.97, 0.90, f'ρ = {rho:.3f}  P = {pv:.1e}',
         transform=ax2.transAxes, ha='right', va='top', fontsize=6.5, color=N_DARK,
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#dddddd', alpha=0.85, lw=0.4))
ax2.set_xlabel(r'Sample size $n$')
ax2.set_ylabel(r'$\Delta$ (CAGate $-$ NOTEARS, edges)')
plabel(ax2, 'b')

save_path = os.path.join(OUT, 'Fig3_HighDim.png')
fig.savefig(save_path, dpi=400, facecolor='white', edgecolor='none')
plt.close(fig)
print(f'[SAVED] Fig3_HighDim.png  ({os.path.getsize(save_path)//1024} KB)')
