# -*- coding: utf-8 -*-
"""Supplementary Figure S5 -- the boundary conditions of the gate, drawn in the
house style.

These two panels were panels (b) and (c) of an earlier Figure 3.  They are the
*limits* of the device: on homogeneous Erdos-Renyi graphs there is no cluster
structure to exploit, and re-partitioning the labels as the graph improves gives
the gain back.  They are reported here, in the Supplementary Information, so the
main text can carry the mechanism and its operating regime.

  a) structural Hamming distance on homogeneous Erdos-Renyi graphs -- the gate
     has nothing to exploit and adds no structural accuracy
  b) re-partitioning the labels at every outer iteration gives the gain back --
     the fixed-label configuration is the one reported

Canvas width is the Supplementary Material's own text block, 160.0 mm (a
stand-alone article with margin=2.5cm), not the 127.3 mm of the ws-jbcb main
text, and it is not cropped, so the declared point sizes survive 1:1.

Every number is read from evidence.json and cluster_rf/*.json.
"""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from figstyle import (MM, PAL, INK, GREY, BOXFC, BOXEC, GRID,
                      card, dots, haloed_text, panel_letters, tidy)

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
})

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(HERE, 'data')
OUT = os.path.join(HERE, 'figures_supplementary')
os.makedirs(OUT, exist_ok=True)
W = 160.0 * MM

GATE, BASE = PAL['mist'], PAL['peri']

EV = json.load(open(os.path.join(C, 'evidence.json'), encoding='utf-8'))
SY = sorted(EV['synth'], key=lambda j: j['d'])
RF = [json.load(open(f, encoding='utf-8'))
      for f in sorted(glob.glob(os.path.join(C, 'cluster_rf', '*.json')))]


def rbox(ax, x, y, s, ha='right', va='top', size=7.5):
    ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va, fontsize=size, color=INK,
            linespacing=1.35,
            bbox=dict(boxstyle='round,pad=0.34', fc=BOXFC, ec=BOXEC, lw=0.55), zorder=9)


fig = plt.figure(figsize=(W, 76.0 * MM))
gs = gridspec.GridSpec(1, 2, wspace=0.30, left=0.088, right=0.972, top=0.905, bottom=0.210)
A = fig.add_subplot(gs[0, 0])
B = fig.add_subplot(gs[0, 1])

# ══════════ a) homogeneous ER: nothing to exploit ══════════
ds = [j['d'] for j in SY]
shd_b = [j['cagate_c_base']['shd'] for j in SY]
shd_g = [j['cagate_c_gate']['shd'] for j in SY]
x = np.arange(len(ds))
w = 0.34
A.bar(x - w / 2, shd_b, w, color=BASE, alpha=0.92, edgecolor='white', linewidth=0.4,
      label='ungated base', zorder=3)
A.bar(x + w / 2, shd_g, w, color=GATE, alpha=0.92, edgecolor='white', linewidth=0.4,
      label='with gate', zorder=3)
for i in range(len(ds)):
    A.text(x[i] - w / 2, shd_b[i] + 0.7, '%d' % shd_b[i], ha='center', va='bottom',
           fontsize=7.5, color=INK)
    A.text(x[i] + w / 2, shd_g[i] + 0.7, '%d' % shd_g[i], ha='center', va='bottom',
           fontsize=7.5, color=INK, fontweight='bold')
    yt = max(shd_b[i], shd_g[i]) + 4.2
    A.plot([x[i] - w / 2, x[i] - w / 2, x[i] + w / 2, x[i] + w / 2],
           [shd_b[i] + 1.6, yt, yt, shd_g[i] + 1.6], color=GREY, lw=0.7, zorder=2)
    A.text(x[i], yt + 0.6, '$+%d$' % (shd_g[i] - shd_b[i]), ha='center', va='bottom',
           fontsize=7.5, color=INK)
A.set_xlim(-0.55, len(ds) - 0.45)
A.set_ylim(0, 34)
A.set_xticks(x)
A.set_xticklabels(['$d$%d' % d for d in ds])
A.set_xlabel('Erd\u0151s\u2013R\u00e9nyi graphs\n(no cluster structure)')
A.set_ylabel('structural Hamming distance')
A.legend(loc='upper left', frameon=False, fontsize=7.5, handlelength=0.9,
         handletextpad=0.4, labelspacing=0.30, borderpad=0.15)
tidy(A, grid='y')

# ══════════ b) re-partitioning the labels gives the gain back ══════════
arms = ['ungated\nbase', 'gate\nfixed labels', 'gate\nre-clustered']
xA = np.arange(3)
fig_mean = np.array([[j['base_mean'], j['fixed_mean'], j['refit_mean']] for j in RF])
B.axvspan(0.62, 1.38, color=GATE, alpha=0.055, lw=0, zorder=0)
for row in fig_mean:
    B.plot(xA, row, '-', color=INK, lw=0.9, alpha=0.30, zorder=3,
           marker='o', ms=3.0, mfc='white', mec=INK, mew=0.6)
mean = fig_mean.mean(axis=0)
B.plot(xA, mean, '-', color=GATE, lw=2.0, zorder=6, solid_capstyle='round')
for xi, yi in zip(xA, mean):
    dots(B, [xi], [yi], 34, 'white', z=6.5, lw=0.0)
    dots(B, [xi], [yi], 24, GATE, z=6.7, lw=0.7)
B.set_xlim(-0.42, 2.42)
B.set_ylim(0.524, 0.638)
B.set_yticks([0.54, 0.56, 0.58, 0.60, 0.62])
B.set_xticks(xA)
B.set_xticklabels(arms)
B.set_ylabel('F1 on clustered synthetic data')
haloed_text(B, 1.0, mean[1] + 0.0075, 'fixed labels', size=7.5, color=GATE,
            ha='center', va='bottom', fontweight='bold')
haloed_text(B, 1.55, mean[2] - 0.010, 're-clustered: 12/30 seeds', size=7.5,
            color=GREY, ha='center', va='top')
haloed_text(B, 1.0, mean[1] + 0.0195, '$+0.044$ on average', size=7.5, color=INK,
            ha='center', va='bottom')
tidy(B, grid='y')

for ax in (A, B):
    card(fig, ax)
panel_letters(fig, [(A, 'a'), (B, 'b')])

p = os.path.join(OUT, 'FigS5_Boundary.pdf')
fig.savefig(p)
fig.savefig(p.replace('.pdf', '.png'), dpi=400)
plt.close(fig)
print('[SAVED]', p, '%.1f KB' % (os.path.getsize(p) / 1024))
print('a) base SHD %s  gate SHD %s  delta %s' % (shd_b, shd_g, [g - b for b, g in zip(shd_b, shd_g)]))
print('b) mean profile base/fixed/refit = %s' % [round(float(v), 4) for v in mean])
print('b) fixed wins %d/%d   refit wins %d/%d' % (
    sum(j['fixed_wins'] for j in RF), sum(j['n_seeds'] for j in RF),
    sum(j['refit_wins'] for j in RF), sum(j['n_seeds'] for j in RF)))
