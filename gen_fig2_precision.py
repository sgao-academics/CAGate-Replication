# -*- coding: utf-8 -*-
"""
NEW Fig. 2 (main text) — research-first version.

The old Fig. 2 sold "CAGate recovers more edges" (Delta = +158).  That claim is
withdrawn: under the corrected solver the gate recovers FEWER edges, but a
larger fraction of them are independently supported.

Panels
  (a) Edges per cancer, three arms        -> base == NOTEARS in magnitude; gate lower
  (b) STRING edge-level support, three arms + random  -> gate > base > NOTEARS
  (c) Paired gate - NOTEARS support, per cancer, sorted  -> 32/33 positive
  (d) TRRUST curated-pair enrichment (either direction), three arms + random

Canvas is 5.0 in wide (= JBCB text width), so an 8 pt label prints at 8 pt.
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(HERE, 'data')
OUT = os.path.join(HERE, 'figures')
os.makedirs(OUT, exist_ok=True)

C_BASE, C_GATE, C_NOT, C_RAND = '#0072B2', '#D55E00', '#009E73', '#999999'
C_TEXT, C_GRID = '#333333', '#E0E0E0'
ARMS = ['base', 'gate', 'notears']
ARM_LBL = {'base': 'CAGate base', 'gate': 'CAGate gate', 'notears': 'NOTEARS'}
ARM_C = {'base': C_BASE, 'gate': C_GATE, 'notears': C_NOT}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7.5, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8, 'legend.fontsize': 6.5,
    'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 2, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
})

EV = json.load(open(os.path.join(C, 'evidence.json'), encoding='utf-8'))
rows = EV['per_cancer']
T = EV['tests']
dv = json.load(open(os.path.join(C, 'dirval_results.json'), encoding='utf-8'))
cancers = sorted(dv)


def plabel(ax, s):
    ax.text(-0.19, 1.05, s, transform=ax.transAxes, fontsize=9.5,
            fontweight='bold', va='bottom', ha='left', color=C_TEXT)


def strip(ax, groups, colors, labels, ylabel, rng_seed=0):
    """Box + jittered points for a list of 1-D arrays."""
    rng = np.random.default_rng(rng_seed)
    bp = ax.boxplot(groups, positions=range(1, len(groups) + 1), widths=0.5,
                    showfliers=False, patch_artist=True, medianprops=dict(color=C_TEXT, lw=1.1),
                    boxprops=dict(lw=0.5), whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5))
    for p, c in zip(bp['boxes'], colors):
        p.set_facecolor(c); p.set_alpha(0.16); p.set_edgecolor(c)
    for i, (g, c) in enumerate(zip(groups, colors), start=1):
        j = rng.normal(0, 0.055, len(g))
        ax.scatter(np.full(len(g), i) + j, g, s=7, c=c, alpha=0.75,
                   edgecolor='white', linewidth=0.2, zorder=5)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)


fig = plt.figure(figsize=(5.0, 4.45))
gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.46,
                      left=0.10, right=0.98, bottom=0.09, top=0.955)

# ── (a) edges per cancer, three arms ──
ax = fig.add_subplot(gs[0, 0])
groups = [[r[k] for r in rows] for k in ARMS]
strip(ax, groups, [ARM_C[k] for k in ARMS], [ARM_LBL[k] for k in ARMS],
      'Edges ($|W|>0.3$)')
ax.set_ylim(-4, 88)
note = 'base $-$ NOTEARS $%+.1f$ edges' % T['base - NOTEARS (edges)']['mean']
ax.text(0.97, 0.03, note, transform=ax.transAxes, ha='right', va='bottom',
        fontsize=6.3, color=C_TEXT,
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=C_GRID, lw=0.4, alpha=0.95))
plabel(ax, 'a')

# ── (b) STRING support by arm ──
ax = fig.add_subplot(gs[0, 1])
S = {'base': 's_base', 'gate': 's_gate', 'notears': 's_notears'}
groups = [[r[S[k]] for r in rows] for k in ARMS]
strip(ax, groups, [ARM_C[k] for k in ARMS], [ARM_LBL[k] for k in ARMS],
      'STRING-supported edges (%)')
ax.set_ylim(-3, 60)
ax.axhline(np.mean([r['s_rand'] for r in rows]), ls='--', lw=0.9,
           color=C_RAND, zorder=1)
ax.text(0.6, np.mean([r['s_rand'] for r in rows]) + 1.4, 'random',
        ha='left', va='bottom', fontsize=6.2, color=C_RAND)
ax.text(0.97, 0.96, 'gate $-$ NOTEARS\n$+%.2f$ pp ($P=%.0e$)' % (
    T['gate - NOTEARS (STRING)']['mean'], T['gate - NOTEARS (STRING)']['p']),
    transform=ax.transAxes, ha='right', va='top', fontsize=6.3, color=C_TEXT,
    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=C_GRID, lw=0.4, alpha=0.95))
plabel(ax, 'b')

# ── (c) paired gate - NOTEARS, per cancer ──
ax = fig.add_subplot(gs[1, 0])
dif = np.array([r['s_gate'] - r['s_notears'] for r in rows])
order = np.argsort(dif)
cs = [cancers[i] for i in order]
v = dif[order]
cols = [C_GATE if x > 0 else C_BASE for x in v]
ax.barh(range(len(v)), v, color=cols, height=0.7, alpha=0.85,
        edgecolor='white', linewidth=0.2)
ax.axvline(0, color=C_TEXT, lw=0.7, alpha=0.5)
ax.set_yticks([0, len(v) // 2, len(v) - 1])
ax.set_yticklabels([cs[0], cs[len(v) // 2], cs[-1]])
ax.set_ylim(-0.8, len(v) - 0.2)
ax.set_xlabel('STRING support, gate $-$ NOTEARS (pp)')
npos = int((dif > 0).sum())
ax.text(0.97, 0.06, '%d / %d cancers positive' % (npos, len(dif)),
        transform=ax.transAxes, ha='right', va='bottom', fontsize=6.3, color=C_TEXT)
plabel(ax, 'c')

# ── (d) TRRUST either-direction enrichment ──
ax = fig.add_subplot(gs[1, 1])
gr = [[dv[c][k]['either_pct'] for c in cancers] for k in ARMS]
rnd = [dv[c]['base']['rnd_pct'] for c in cancers]
strip(ax, gr, [ARM_C[k] for k in ARMS], [ARM_LBL[k] for k in ARMS],
      'In TRRUST (%)', rng_seed=3)
ax.set_ylim(-1.5, 27)
ax.axhline(np.mean(rnd), ls='--', lw=0.9, color=C_RAND, zorder=1)
ax.text(0.6, np.mean(rnd) + 0.5, 'random', ha='left', va='bottom',
        fontsize=6.2, color=C_RAND)
enr = np.mean(gr[1]) / np.mean(rnd)
ax.text(0.97, 0.96, 'gate %.1f$\\times$ random\n(either direction)' % enr,
        transform=ax.transAxes, ha='right', va='top', fontsize=6.3, color=C_TEXT,
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=C_GRID, lw=0.4, alpha=0.95))
plabel(ax, 'd')

p = os.path.join(OUT, 'Fig2_Pancancer.pdf')
fig.savefig(p, facecolor='white', edgecolor='none')
fig.savefig(p.replace('.pdf', '.png'), dpi=400, facecolor='white', edgecolor='none')
plt.close(fig)
print('[SAVED]', p, '%.1f KB' % (os.path.getsize(p) / 1024))
