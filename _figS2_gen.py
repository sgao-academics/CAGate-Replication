# -*- coding: utf-8 -*-
"""Figure S3 -- behaviour and robustness of the residual-contrast gate.

Figure 2 answers "is the gate's improvement real?"  Figure 3 answers "where does
it act?"  This figure answers the two questions left over: does the gate shrink
the graph rather than inflate it (a), and does the conclusion depend on its only
hyperparameter (b).

Chart-type discipline: a histogram and a dot plot with a reference line -- two
idioms not used anywhere else in the paper.

Canvas = 160.0 mm -- the Supplementary Material's own text block (a stand-alone
article with margin=2.5cm), not the 127.3 mm of the ws-jbcb main text -- and it
is not cropped, so the declared point sizes survive 1:1.
(a) reads the 33 pan-cancer runs in mega33_results/; (b) reads alpha_sweep.json.
"""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from figstyle import (MM, PAL, INK, GREY, BOXFC, BOXEC,
                      card, dots, panel_letters, tidy)

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
})

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(HERE, 'data')          # all inputs live here
OUT = os.path.join(HERE, 'figures_supplementary')   # figure output
os.makedirs(OUT, exist_ok=True)
W = 160.0 * MM                                   # == \textwidth of the SI

# ---- (a) gate minus base, one value per cancer type -------------------------
DELTA = np.array([json.load(open(f, encoding='utf-8'))['gate']['edges']
                  - json.load(open(f, encoding='utf-8'))['base']['edges']
                  for f in sorted(glob.glob(os.path.join(C, 'mega33_results', '*.json')))],
                 dtype=float)
# ---- (b) sensitivity to alpha ----------------------------------------------
AS = json.load(open(os.path.join(C, 'alpha_sweep.json'), encoding='utf-8'))
ungated = AS['base']['edges']
al = np.array([r['alpha'] for r in AS['rows']], float)
ed = np.array([r['edges'] for r in AS['rows']], float)

fig = plt.figure(figsize=(W, 73.0 * MM))
gs = gridspec.GridSpec(1, 2, wspace=0.34, left=0.095, right=0.985, top=0.868, bottom=0.160)
A = fig.add_subplot(gs[0, 0])
B = fig.add_subplot(gs[0, 1])

# ══════════ a) the gate removes edges, in every cancer type ══════════
bins = np.arange(-26, 3, 4)
cnt, _, _ = A.hist(DELTA, bins=bins, color=PAL['mist'], edgecolor='white',
                   linewidth=0.6, zorder=3)
A.axvline(0, color=GREY, ls=(0, (3.0, 2.4)), lw=0.9, zorder=1)
A.axvline(DELTA.mean(), color=PAL['indigo'], lw=1.1, zorder=4)
A.set_xlim(-27, 4.5)
A.set_ylim(0, cnt.max() * 1.62)
A.set_yticks([0, 5, 10, 15, 20])          # explicit, so no tick sits outside the view
A.set_xlabel('edges removed by the gate  (gate $-$ base)')
A.set_ylabel('cancer types')
A.text(DELTA.mean() - 0.5, cnt.max() * 1.52, 'mean $%.1f$' % DELTA.mean(),
       ha='right', va='top', fontsize=7.5, color=INK)
n_neg = int((DELTA < 0).sum())
A.text(0.975, 0.965, 'fewer edges in\n%d of %d cancers' % (n_neg, len(DELTA)),
       transform=A.transAxes, ha='right', va='top', fontsize=7.5, color=INK,
       linespacing=1.35,
       bbox=dict(boxstyle='round,pad=0.34', fc=BOXFC, ec=BOXEC, lw=0.55), zorder=9)
tidy(A, grid='y')

# ══════════ b) one hyperparameter, and it does not matter ══════════
B.set_xscale('log')
B.set_xlim(0.076, 2.75)
B.set_ylim(38, 74)
B.set_yticks([40, 50, 60, 70])            # explicit, so no tick sits outside the view
B.axhspan(50, 52, color=PAL['mist'], alpha=0.11, lw=0, zorder=0)
B.axhline(ungated, color=GREY, ls=(0, (3.0, 2.4)), lw=0.9, zorder=1)
dots(B, al, ed, 32, PAL['mist'], z=5.5, lw=0.6, ec='white')
B.plot([0.5], [50.0], 'o', ms=11.5, mfc='none', mec=INK, mew=1.0, zorder=6)
B.set_xticks([0.1, 0.25, 0.5, 1.0, 2.0])
B.set_xticklabels(['0.10', '0.25', '0.50', '1.0', '2.0'])
B.minorticks_off()
B.set_xlabel('gate sensitivity $\\alpha$  (log scale)')
B.set_ylabel('edges recovered')
B.text(2.62, ungated + 1.2, 'ungated (%d)' % ungated, ha='right', va='bottom',
       fontsize=7.5, color=GREY)
B.text(0.085, 53.6, '%d\u2013%d edges across the sweep' % (ed.min(), ed.max()),
       ha='left', va='bottom', fontsize=7.5, color=INK)
B.annotate('default $\\alpha=0.50$', xy=(0.5, 50.0), xytext=(0.5, 43.5),
           ha='center', va='top', fontsize=7.5, color=GREY,
           arrowprops=dict(arrowstyle='-', lw=0.6, color=GREY, shrinkA=1, shrinkB=3))
tidy(B, grid='y')

for ax in (A, B):
    card(fig, ax)
panel_letters(fig, [(A, 'a'), (B, 'b')])

p = os.path.join(OUT, 'FigS2_Alpha.pdf')
fig.savefig(p)
fig.savefig(p.replace('.pdf', '.png'), dpi=400)
plt.close(fig)
print('[SAVED]', p, '%.1f KB' % (os.path.getsize(p) / 1024))
print('a) N=%d  mean=%.2f  min=%d  max=%d  n_neg=%d  hist=%s'
      % (len(DELTA), DELTA.mean(), DELTA.min(), DELTA.max(), n_neg, list(cnt)))
print('b) ungated %d | alpha %s -> edges %s' % (ungated, list(al), list(ed)))
