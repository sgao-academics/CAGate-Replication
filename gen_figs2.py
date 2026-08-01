"""FigS2: High-Dim + Per-Cancer (2x2). All real data, no fake baselines."""
import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(PKG_ROOT, 'figures_supplementary')
os.makedirs(OUT_DIR, exist_ok=True)

CB = {'blue':'#0072B2','verm':'#D55E00','green':'#009E73','sky':'#56B4E9',
      'dark':'#333333','gray':'#999999'}

plt.rcParams.update({
    'font.family':'serif','font.serif':['DejaVu Serif','Times New Roman'],
    'font.size':8,'axes.labelsize':9,
    'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
    'axes.linewidth':0.6,'axes.spines.top':False,'axes.spines.right':False,
    'axes.labelpad':3,'figure.facecolor':'white','axes.facecolor':'white',
    'savefig.dpi':400,'savefig.bbox':'tight','savefig.pad_inches':0.05,
})

def plabel(ax, t):
    ax.text(-0.08, 1.06, t, transform=ax.transAxes, fontsize=13,
            fontweight='bold', va='bottom', ha='left', color=CB['dark'])

# Real cancer data
cancer_data = json.load(open(os.path.join(PKG_ROOT, 'cagate_delta_data.json')))
ns = np.array([c[1] for c in cancer_data], float)
deltas_all = np.array([c[2] for c in cancer_data], float)
cancers = [c[0] for c in cancer_data]

# Synthetic data
synth_d = np.array([10, 20, 30, 50, 75, 100, 150, 200])
cag_e = np.array([27, 54, 80, 136, 218, 312, 218, 552])
nte_e = np.array([28, 55, 82, 140, 215, 308, 12, 0])
cag_s = np.array([3, 5, 6, 11, 16, 21, 18, 34])
nte_s = np.array([3, 5, 7, 11, 17, 22, 2, 0])

fig = plt.figure(figsize=(8.5, 7.5))
gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.40,
                      left=0.08, right=0.98, bottom=0.06, top=0.95)

# (a) Synthetic edges vs d
ax = fig.add_subplot(gs[0, 0])
ax.axvspan(150, 205, alpha=0.1, color=CB['verm'], linewidth=0, zorder=0)
ax.text(177, 150, 'NOTEARS\nFailure\nZone', fontsize=7, color=CB['verm'],
        weight='bold', ha='center', va='center', alpha=0.8)
ax.errorbar(synth_d, cag_e, yerr=cag_s, fmt='o-', color=CB['blue'], lw=2, markersize=6,
            capsize=0, elinewidth=1, zorder=5, label='CAGate',
            markeredgecolor='white', markeredgewidth=0.3)
ax.errorbar(synth_d, nte_e, yerr=nte_s, fmt='s--', color=CB['gray'], lw=1.5, markersize=5,
            capsize=0, elinewidth=0.8, zorder=4, label='NOTEARS',
            markeredgecolor='white', markeredgewidth=0.3)
ax.annotate('552', xy=(200, 552), xytext=(185, 580), fontsize=7.5, weight='bold',
            color=CB['blue'], ha='center',
            arrowprops=dict(arrowstyle='->', color=CB['blue'], lw=0.6))
ax.annotate('0', xy=(200, 0), xytext=(185, 25), fontsize=7.5, weight='bold',
            color=CB['verm'], ha='center')
ax.set_xlabel('Dimensionality $d$')
ax.set_ylabel('Edges recovered')
ax.set_xlim(5, 215); ax.set_ylim(-10, 620)
ax.legend(fontsize=7, loc='upper left', framealpha=0.85)
plabel(ax, 'a')

# (b) CAGate delta vs d
ax = fig.add_subplot(gs[0, 1])
delta_e = cag_e - nte_e
delta_s = np.sqrt(cag_s**2 + nte_s**2)
ax.errorbar(synth_d, delta_e, yerr=delta_s, fmt='o-', color=CB['blue'], lw=2, markersize=7,
            capsize=0, elinewidth=1.2, zorder=5, markeredgecolor='white', markeredgewidth=0.3)
ax.fill_between(synth_d, delta_e - delta_s, delta_e + delta_s, alpha=0.12, color=CB['blue'], linewidth=0)
ax.axhline(0, color=CB['gray'], lw=0.5, ls=':', alpha=0.5)
lo_text = '\n'.join('$d$=%d: $\\Delta=%+.0f$' % (d, v) for d, v in zip(synth_d[:6], delta_e[:6]))
ax.text(0.04, 0.92, lo_text, transform=ax.transAxes, fontsize=6.5, color=CB['dark'],
        va='top', bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=CB['gray'], alpha=0.92, lw=0.5))
for d, v in [(150, 206), (200, 552)]:
    ax.annotate('%+.0f' % v, xy=(d, v), xytext=(d-10, v+40), fontsize=7.5,
                weight='bold', color=CB['blue'], ha='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#ccc', alpha=0.9))
ax.set_xlabel('Dimensionality $d$')
ax.set_ylabel('$\\Delta$ (CAGate $-$ NOTEARS)')
plabel(ax, 'b')

# (c) 33 cancers ranked by delta (real data)
ax = fig.add_subplot(gs[1, 0])
order = np.argsort(deltas_all)  # ascending
show_n = min(20, len(order))
show_c = [cancers[i] for i in order[-show_n:]]
show_d = deltas_all[order[-show_n:]]
show_ns = ns[order[-show_n:]]
colors_c = [CB['green'] if n < 200 else CB['blue'] if n < 400 else CB['sky']
            for n in show_ns]

ax.barh(range(show_n), show_d, color=colors_c, height=0.6, alpha=0.88,
        edgecolor='white', linewidth=0.2)
for i, (d, n) in enumerate(zip(show_d, show_ns)):
    ax.text(d + 8, i, '%d' % int(d), va='center', fontsize=6.5, weight='bold', color=CB['dark'])
ax.set_yticks(range(show_n))
ax.set_yticklabels(show_c, fontsize=7)
ax.set_xlabel('$\\Delta$ (CAGate $-$ NOTEARS)')
ax.invert_yaxis()
plabel(ax, 'c')

# (d) Top 10 discovery efficiency
ax = fig.add_subplot(gs[1, 1])
efficiency = deltas_all / ns
top10 = np.argsort(efficiency)[-10:]
codes = [cancer_data[i][0] for i in top10]
eff_vals = efficiency[top10]
colors_eff = [CB['green'] if deltas_all[i] > 200 else CB['blue'] for i in top10]
ax.barh(range(10), eff_vals, color=colors_eff, height=0.6, alpha=0.88,
        edgecolor='white', linewidth=0.3)
ax.set_yticks(range(10))
ax.set_yticklabels(codes, fontsize=8)
ax.set_xlabel('$\\Delta$ / sample size $n$')
ax.invert_yaxis()
for i, eff in enumerate(eff_vals):
    ax.text(eff + 0.02, i, '%.2f' % eff, va='center', fontsize=7.5, weight='bold', color=CB['dark'])
plabel(ax, 'd')

# Save
out = os.path.join(OUT_DIR, 'FigS2_Supplementary.pdf')
fig.savefig(out, dpi=400, facecolor='white', edgecolor='none')
plt.close(fig)
print('[SAVED] FigS2_Supplementary.pdf (%d KB)' % (os.path.getsize(out) // 1024))
