# -*- coding: utf-8 -*-
"""Figure 1, drawn in the house style (作图心得与模板/00_核心规范).

Canvas width is the manuscript's own text block, 127.3 mm (361 pt, measured from
ws-jbcb), not the library's 174 mm: Fig. 1 is placed at \\textwidth, so drawing at
174 mm would scale every 8 pt label down to 5.8 pt and break WSPC's 8 pt floor.

a) subpopulations differ in the dispersion of their residuals
b) the residual-contrast gate maps that dispersion onto a weight
c) the gated loss therefore concentrates on the coherent subpopulations

Every number comes from the paper's own clustered generator and its own solver.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from figstyle import (MM, PAL, INK, GREY, TRACK, BOXFC, BOXEC,
                      card, dots, capsule, haloed_text, panel_letters, tidy)
from cagate import gen_cluster, solve_cagate_lbfgsb

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
})

W = 127.3 * MM                       # == \textwidth of ws-jbcb

ALPHA, D, NC, SP = 0.5, 15, 5, 100
X, Wt, cid = gen_cluster(d=D, nc=NC, sp=SP, seed=0)
res = solve_cagate_lbfgsb(X, cid=cid, use_gate=True, l1=0.01, gate_alpha=ALPHA)
R = X - X @ res['W']
e = (R ** 2).mean(axis=1)
sig = np.array([e[cid == k].std(ddof=1) for k in range(NC)])
mu = np.array([e[cid == k].mean() for k in range(NC)])
sbar = float(np.median(sig))
gam = 1.0 / (1.0 + np.exp(-ALPHA * (sbar - sig)))
rel = sig / sbar
order = np.argsort(sig)
rank = np.empty(NC, int); rank[order] = np.arange(NC)
COL = [PAL['mist'], PAL['peri'], PAL['moss'], PAL['orchid'], PAL['violet']]
LBL = ['C%d' % i for i in reversed(range(1, NC + 1))]
print('rel', np.round(rel, 3).tolist(), 'gamma', np.round(gam, 4).tolist())

# ── canvas: two rows, the dispersion strip across the top ──────────────────
fig = plt.figure(figsize=(W, W * 0.70))
gs = gridspec.GridSpec(2, 2, height_ratios=[0.80, 1.0], width_ratios=[1.14, 1.0],
                       hspace=0.62, wspace=0.42, left=0.083, right=0.985,
                       top=0.935, bottom=0.115)
A = fig.add_subplot(gs[0, :])
B = fig.add_subplot(gs[1, 0])
C = fig.add_subplot(gs[1, 1])

# ══════════ a) residual scatter differs across subpopulations ══════════
rng = np.random.RandomState(0)
A.set_xscale('log')
A.set_xlim(float(np.percentile(e, 0.2)) * 0.55, float(np.percentile(e, 99.8)) * 3.0)
A.set_ylim(-0.78, NC - 0.22)
for i in range(NC):
    k = order[i]
    y = NC - 1 - i
    ek = e[cid == k]
    m, s = mu[k], sig[k]
    A.barh(y, 2 * s, left=max(m - s, 1e-3), height=0.60, color=COL[i],
           alpha=0.17, edgecolor='none', zorder=2)            # +/- 1 sd = sigma_k
    A.scatter(ek, y + (rng.rand(len(ek)) - 0.5) * 0.44, s=4.5, c=COL[i],
              alpha=0.55, edgecolors='none', zorder=3)
    A.plot([max(m - s, 1e-3), m + s], [y, y], color=COL[i], lw=2.2, zorder=5,
           solid_capstyle='round')
    A.plot([m], [y], marker='|', ms=6, mew=1.5, color=INK, zorder=6)
A.set_xticks([0.5, 1, 2, 5, 10, 20])
A.set_xticklabels(['0.5', '1', '2', '5', '10', '20'])
A.set_yticks(range(NC)); A.set_yticklabels(LBL, fontsize=7.5)
A.set_xlabel(r'residual energy  $e_i=\|x_i-x_iW\|_2^2/d$')
A.set_ylabel('subpopulation')
tidy(A, grid='x')
A.text(0.995, 0.90, 'shaded band $=\\pm1$ sd $=\\sigma_k$', transform=A.transAxes,
       ha='right', va='center', fontsize=7.5, color=GREY, zorder=7)

# ══════════ b) the gate: dispersion -> weight ══════════
B.set_xlim(0.34, 2.66); B.set_ylim(0.15, 0.76)
B.axvspan(0.34, 1.0, color=PAL['mist'], alpha=0.055, lw=0, zorder=1)
B.axvspan(1.0, 2.66, color=PAL['violet'], alpha=0.055, lw=0, zorder=1)
xx = np.linspace(0.34, 2.66, 300)
B.plot(xx, 1.0 / (1.0 + np.exp(-ALPHA * sbar * (1.0 - xx))), color=INK, lw=1.6, zorder=4)
B.axhline(0.5, color=GREY, ls=(0, (1.5, 2)), lw=0.8, zorder=2)
B.axvline(1.0, color=GREY, ls=(0, (1.5, 2)), lw=0.8, zorder=2)
_off = [(6, 2, 'left'), (6, 2, 'left'), (6, 2, 'left'), (-6, 5, 'right'), (7, 6, 'left')]
for i in range(NC):
    k = order[i]
    dots(B, [rel[k]], [gam[k]], 27, COL[i], z=5, lw=0.5)
    B.annotate('C%d' % (i + 1), xy=(rel[k], gam[k]), xytext=(_off[i][0], _off[i][1]),
               textcoords='offset points', fontsize=7.5, color=COL[i],
               fontweight='bold', ha=_off[i][2], va='center', zorder=6)
B.text(0.42, 0.755, 'coherent\nup-weighted', fontsize=7.5, color=PAL['mist'],
       ha='left', va='top', fontweight='bold', zorder=6)
B.text(2.63, 0.755, 'dispersed\nattenuated', fontsize=7.5, color=PAL['violet'],
       ha='right', va='top', fontweight='bold', zorder=6)
B.text(2.63, 0.515, r'$\gamma=1/2$', fontsize=7.5, color=GREY, ha='right', va='bottom')
B.text(1.05, 0.300, r'median $\bar{\sigma}$', fontsize=7.5, color=GREY, ha='left', va='bottom')
B.text(0.03, 0.055, r'$\gamma_k=\left[\,1+e^{-\alpha(\bar{\sigma}-\sigma_k)}\,\right]^{-1}$',
       transform=B.transAxes, ha='left', va='bottom', fontsize=8.5, color=INK,
       bbox=dict(boxstyle='round,pad=0.38', fc=BOXFC, ec=BOXEC, lw=0.6), zorder=7)
B.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])
B.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
B.set_xlabel(r'relative residual dispersion  $\sigma_k/\bar{\sigma}$')
B.set_ylabel(r'gate weight  $\gamma_k$')
tidy(B, grid=None)

# ══════════ c) weight applied to each subpopulation ══════════
C.set_xlim(0.0, 1.18); C.set_ylim(-0.66, NC - 0.34)
for i in range(NC):
    k = order[i]
    y = NC - 1 - i
    capsule(C, 0.0, 1.0, y, 0.56, TRACK, z=2)
    capsule(C, 0.0, float(gam[k]), y, 0.56, COL[i], z=3)
    haloed_text(C, float(gam[k]) + 0.030, y, '%.2f' % gam[k], va='center', ha='left', size=7.5)
C.axvline(0.5, color=GREY, ls=(0, (1.5, 2)), lw=0.8, zorder=2)
C.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
C.set_yticks(range(NC)); C.set_yticklabels(LBL, fontsize=7.5)
C.set_xlabel(r'weight on the subpopulation  $\gamma_k$')
C.text(0.5, -0.56, 'pale track = pooled weight (1.0)', fontsize=7.5, color=GREY,
       ha='center', va='center')
tidy(C, grid=None)

for ax in (A, B, C):
    card(fig, ax)
panel_letters(fig, [(A, 'a'), (B, 'b'), (C, 'c')])

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figures', 'Fig1_Mechanism')
fig.savefig(OUT + '.pdf')
fig.savefig(OUT + '.png', dpi=400)
print('wrote', OUT + '.pdf')
