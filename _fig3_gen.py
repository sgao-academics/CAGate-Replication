# -*- coding: utf-8 -*-
"""Figure 3, drawn in the house style (作图心得与模板/00_核心规范).

Figure 3 carries the *mechanism and its scope*, now that the volume/precision
evidence lives in Figure 2:

  a) the gate raises F1 wherever there is genuine cluster heterogeneity
     (27 of 30 seeds) -- violin + per-seed dots
  b) on homogeneous Erdos-Renyi graphs it never helps -- paired columns
  c) re-partitioning the labels every iteration gives the gain back -- slope

Chart-type discipline: Figure 1 already uses a residual strip, a logistic curve
and capsules; Figure 2 uses scatter, paired scatter and capsules.  So this
figure deliberately uses three further idioms -- a violin, a paired column and a
slope -- and no capsules at all.

Canvas width is the manuscript's own text block, 127.3 mm (361 pt, measured from
ws-jbcb), because Figure 3 is placed at \\textwidth: drawing at the library's
174 mm would scale every label down and break WSPC's 8 pt floor.

Every number comes from the corrected solver's own evidence files.
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
C = os.path.join(HERE, 'data')          # all inputs live here
OUT = os.path.join(HERE, 'figures')   # figure output
os.makedirs(OUT, exist_ok=True)
W = 127.3 * MM                                   # == \textwidth of ws-jbcb

GATE, BASE = PAL['mist'], PAL['peri']            # same job colours as Figs. 1-2

EV = json.load(open(os.path.join(C, 'evidence.json'), encoding='utf-8'))
CL = EV['cluster']                               # clustered synthetic, 3 configs
SY = sorted(EV['synth'], key=lambda j: j['d'])   # homogeneous ER sweep
RF = [json.load(open(f, encoding='utf-8'))
      for f in sorted(glob.glob(os.path.join(C, 'cluster_rf', '*.json')))]


def rbox(ax, x, y, s, ha='right', va='top', size=7.5):
    """Rounded annotation container -- the house idiom for a stat callout."""
    ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va, fontsize=size, color=INK,
            linespacing=1.35,
            bbox=dict(boxstyle='round,pad=0.34', fc=BOXFC, ec=BOXEC, lw=0.55), zorder=9)


fig = plt.figure(figsize=(W, 97.0 * MM))
gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 0.78], hspace=0.56, wspace=0.40,
                       left=0.118, right=0.984, top=0.926, bottom=0.102)
A = fig.add_subplot(gs[0, 0])
B = fig.add_subplot(gs[0, 1])
Cc = fig.add_subplot(gs[1, :])

# ══════════ a) clustered synthetic: the gate raises F1 ══════════
deltas = [np.array(j['gate_all'], float) - np.array(j['base_all'], float) for j in CL]
lbls = ['$d$15\n$K$5', '$d$15\n$K$8', '$d$20\n$K$6']
pos = [0, 1, 2]
vp = A.violinplot(deltas, positions=pos, widths=0.74, showextrema=False, showmedians=False)
for body in vp['bodies']:
    body.set_facecolor(GATE)
    body.set_edgecolor(GATE)
    body.set_alpha(0.17)
    body.set_linewidth(0.8)
rng = np.random.default_rng(3)
for i, d in enumerate(deltas):
    A.scatter(pos[i] + rng.uniform(-0.155, 0.155, len(d)), d, s=8, c=GATE,
              alpha=0.80, edgecolors='white', linewidths=0.30, zorder=5)
    m = float(d.mean())
    A.plot([pos[i] - 0.20, pos[i] + 0.20], [m, m], color=INK, lw=1.0, zorder=7)
    dots(A, [pos[i]], [m], 22, 'white', z=7.2, lw=0.0)
    dots(A, [pos[i]], [m], 15, INK, z=7.4, lw=0.5)
A.axhline(0, color=GREY, ls=(0, (3.0, 2.4)), lw=0.9, zorder=1)
A.set_xlim(-0.62, 2.62)
A.set_ylim(-0.062, 0.238)          # top tick sits below the panel letter
A.set_yticks([0.0, 0.05, 0.10, 0.15, 0.20])
A.set_xticks(pos)
A.set_xticklabels(lbls)
A.set_ylabel('$\\Delta$F1 (gate $-$ base)')
tidy(A, grid='y')
wins = sum(int((d > 0).sum()) for d in deltas)
tot = sum(len(d) for d in deltas)
rbox(A, 0.975, 0.985, '%d / %d seeds\nabove zero' % (wins, tot))

# ══════════ b) homogeneous ER: no heterogeneity, no benefit ══════════
ds = [j['d'] for j in SY]
shd_b = [j['cagate_c_base']['shd'] for j in SY]
shd_g = [j['cagate_c_gate']['shd'] for j in SY]
x = np.arange(len(ds))
w = 0.34
B.bar(x - w / 2, shd_b, w, color=BASE, alpha=0.92, edgecolor='white', linewidth=0.4,
      label='ungated base', zorder=3)
B.bar(x + w / 2, shd_g, w, color=GATE, alpha=0.92, edgecolor='white', linewidth=0.4,
      label='with gate', zorder=3)
for i in range(len(ds)):
    B.text(x[i] - w / 2, shd_b[i] + 0.7, '%d' % shd_b[i], ha='center', va='bottom',
           fontsize=7.5, color=INK)
    B.text(x[i] + w / 2, shd_g[i] + 0.7, '%d' % shd_g[i], ha='center', va='bottom',
           fontsize=7.5, color=INK, fontweight='bold')
    # a bracket that reads as one delta per dimension
    yt = max(shd_b[i], shd_g[i]) + 4.2
    B.plot([x[i] - w / 2, x[i] - w / 2, x[i] + w / 2, x[i] + w / 2],
           [shd_b[i] + 1.6, yt, yt, shd_g[i] + 1.6], color=GREY, lw=0.7, zorder=2)
    B.text(x[i], yt + 0.6, '$+%d$' % (shd_g[i] - shd_b[i]), ha='center', va='bottom',
           fontsize=7.5, color=INK)
B.set_xlim(-0.55, len(ds) - 0.45)
B.set_ylim(0, 34)
B.set_xticks(x)
B.set_xticklabels(['$d$%d' % d for d in ds])
B.set_xlabel('Erd\u0151s\u2013R\u00e9nyi graphs\n(no cluster structure)')
B.set_ylabel('structural Hamming distance')
B.legend(loc='upper left', frameon=False, fontsize=7.5, handlelength=0.9,
         handletextpad=0.4, labelspacing=0.30, borderpad=0.15)
tidy(B, grid='y')

# ══════════ c) re-partitioning the labels gives the gain back ══════════
arms = ['ungated\nbase', 'gate\nfixed labels', 'gate\nre-clustered']
xA = np.arange(3)
fig_mean = np.array([[j['base_mean'], j['fixed_mean'], j['refit_mean']] for j in RF])
Cc.axvspan(0.62, 1.38, color=GATE, alpha=0.055, lw=0, zorder=0)
for row in fig_mean:
    Cc.plot(xA, row, '-', color=INK, lw=0.9, alpha=0.30, zorder=3,
            marker='o', ms=3.0, mfc='white', mec=INK, mew=0.6)
mean = fig_mean.mean(axis=0)
Cc.plot(xA, mean, '-', color=GATE, lw=2.0, zorder=6, solid_capstyle='round')
for xi, yi in zip(xA, mean):
    dots(Cc, [xi], [yi], 34, 'white', z=6.5, lw=0.0)
    dots(Cc, [xi], [yi], 24, GATE, z=6.7, lw=0.7)
Cc.set_xlim(-0.42, 2.42)
Cc.set_ylim(0.524, 0.638)          # top tick sits below the panel letter
Cc.set_yticks([0.54, 0.56, 0.58, 0.60, 0.62])
Cc.set_xticks(xA)
Cc.set_xticklabels(arms)
Cc.set_ylabel('F1 on clustered synthetic data')
haloed_text(Cc, 1.0, mean[1] + 0.0075, 'fixed labels', size=7.5, color=GATE,
            ha='center', va='bottom', fontweight='bold')
haloed_text(Cc, 2.0, mean[2] - 0.0075,
            're-clustered: 12/30 seeds', size=7.5, color=GREY, ha='center', va='top')
haloed_text(Cc, 1.0, mean[1] + 0.0195, '$+0.044$ on average', size=7.5, color=INK,
            ha='center', va='bottom')
tidy(Cc, grid='y')

for ax in (A, B, Cc):
    card(fig, ax)
panel_letters(fig, [(A, 'a'), (B, 'b'), (Cc, 'c')])

p = os.path.join(OUT, 'Fig3_Validation.pdf')
fig.savefig(p)
fig.savefig(p.replace('.pdf', '.png'), dpi=400)
plt.close(fig)
print('[SAVED]', p, '%.1f KB' % (os.path.getsize(p) / 1024))
print('a) wins %d/%d  delta means %s' % (wins, tot, [round(float(d.mean()), 4) for d in deltas]))
print('b) base SHD %s  gate SHD %s  delta %s' % (shd_b, shd_g,
                                                 [g - b for b, g in zip(shd_b, shd_g)]))
print('c) mean profile base/fixed/refit = %s' % [round(float(v), 4) for v in mean])
print('c) fixed wins %d/%d   refit wins %d/%d' % (
    sum(j['fixed_wins'] for j in RF), sum(j['n_seeds'] for j in RF),
    sum(j['refit_wins'] for j in RF), sum(j['n_seeds'] for j in RF)))
