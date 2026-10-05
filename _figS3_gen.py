"""Supplementary Figure S3 -- what each panel can measure, and what it can be
tested against, drawn in the house style.

A pair of gene sets can be "described" by a database in two different ways, and
the two panels are not comparable until that difference is removed:

  (a) the share of the 44 850 possible gene pairs about which each source
      carries information at all, per panel -- this is what exposes the KEGG and
      MSigDB ceilings on a KEGG-derived panel;
  (b) the within-panel coherence odds ratios, i.e. the criterion scored against
      evidence that is available on both panels.

Canvas width is the Supplementary Material's own text block, 160.0 mm (a
stand-alone article with margin=2.5cm), not the 127.3 mm of the ws-jbcb main
text, and it is not cropped, so the declared point sizes survive 1:1.  The
idioms used here -- paired horizontal log bars and a two-series grouped odds-
ratio column -- do not recur in Fig. 1-4.

Every number is read from figS34_data.json.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from figstyle import (MM, PAL, INK, GREY, BOXFC, BOXEC, GRID,
                      card, dots, haloed_text, panel_letters, tidy)
from palette import TRACK

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
})

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(HERE, 'data')          # every input lives here
OUT = os.path.join(HERE, 'figures_supplementary')   # figure output
os.makedirs(OUT, exist_ok=True)
W = 160.0 * MM
D = json.load(open(os.path.join(C, 'figS34_data.json'), encoding='utf-8'))

KP, NP = 'KEGG-300', 'non-pathway-300'
COL = {KP: PAL['mist'], NP: PAL['violet']}
CYT, CCY, OTH = PAL['moss'], PAL['violet'], PAL['lilac']
SRCS = ['STRING', 'BioGRID', 'Hallmark', 'KEGG', 'MSigDB']

# ══════════════════════ Figure S4 ══════════════════════
fig = plt.figure(figsize=(W, 86.0 * MM))
gs = gridspec.GridSpec(1, 2, wspace=0.40, left=0.105, right=0.985, top=0.905, bottom=0.175)
A = fig.add_subplot(gs[0, 0])
B = fig.add_subplot(gs[0, 1])

ys = np.arange(len(SRCS))[::-1]
h = 0.30
for k, tag in enumerate([KP, NP]):
    vals = [100.0 * D['panel_pairs'][tag][s] / D['n_possible'] for s in SRCS]
    off = (0.5 - k) * h * 1.06
    A.barh(ys + off, vals, h, color=COL[tag], alpha=0.94, edgecolor='white',
           linewidth=0.4, zorder=3)
    for i, v in enumerate(vals):
        A.text(v * 1.16, ys[i] + off, '%.2f' % v if v < 10 else '%.0f' % v,
               va='center', ha='left', fontsize=7.5, color=INK)
A.set_xscale('log')
A.set_xlim(0.03, 900)
A.set_xticks([0.1, 1, 10, 100])
A.set_xticklabels(['0.1', '1', '10', '100'])
A.set_ylim(-0.62, len(SRCS) - 0.38)
A.set_yticks(ys)
A.set_yticklabels(SRCS)
A.set_xlabel('gene pairs the source can speak about (% of the 44 850 possible)')
tidy(A, grid='x')
A.legend(handles=[Line2D([], [], marker='s', ls='', ms=5.0, mfc=COL[KP], mec=COL[KP],
                         label='KEGG-300 panel'),
                  Line2D([], [], marker='s', ls='', ms=5.0, mfc=COL[NP], mec=COL[NP],
                         label='non-pathway-300 panel')],
         loc='lower right', bbox_to_anchor=(1.0, 1.012), ncol=2, frameon=False,
         fontsize=7.5, handlelength=0.8, handletextpad=0.35, labelspacing=0.28,
         borderpad=0.0, columnspacing=1.6)
haloed_text(A, 850, 3.50, 'the KEGG panel saturates\nKEGG and MSigDB', size=7.5,
            color=INK, ha='right', va='center')

co = D['coherence']
KEY = [('STRING', 'STRING'), ('BioGRID', 'BioGRID'),
       ('KEGG co-membership', 'KEGG'), ('GO co-annotation', 'GO')]
x = np.arange(len(KEY))
w = 0.34
for k, tag in enumerate([KP, NP]):
    vals, sig = [], []
    for key, _ in KEY:
        d = co[tag].get(key)
        vals.append(d['OR'] if (d and d['OR'] == d['OR']) else np.nan)
        sig.append(bool(d and d['OR'] == d['OR'] and d['p'] < 0.05))
    xk = x + (k - 0.5) * w * 1.04
    ok = [i for i in range(len(vals)) if vals[i] == vals[i]]
    B.bar([xk[i] for i in ok], [vals[i] for i in ok], w, color=COL[tag], alpha=0.94,
          edgecolor='white', linewidth=0.4, zorder=3)
    for i in ok:
        if not sig[i]:
            B.bar([xk[i]], [vals[i]], w, color='white', edgecolor=COL[tag],
                  linewidth=0.9, zorder=4)
        B.text(xk[i], vals[i] * 1.10, '%.1f' % vals[i], ha='center', va='bottom',
               fontsize=7.5, color=INK)
B.text(x[2] - 0.5 * w, 0.90, 'n/a', ha='center', va='bottom', fontsize=7.5, color=GREY)
B.axhline(1.0, color=GREY, ls=(0, (3.0, 2.4)), lw=0.9, zorder=1)
B.set_yscale('log')
B.set_ylim(0.72, 42)
B.set_yticks([1, 2, 5, 10, 20])
B.set_yticklabels(['1', '2', '5', '10', '20'])
B.set_xlim(-0.60, len(KEY) - 0.40)
B.set_xticks(x)
B.set_xticklabels([lb for _, lb in KEY])
B.set_xlabel('source (KEGG/GO: shared membership)')
B.set_ylabel('odds ratio, consensus vs rest')
tidy(B, grid='y')
B.legend(handles=[Line2D([], [], marker='s', ls='', ms=5.0, mfc=COL[KP], mec=COL[KP],
                         label='KEGG-300 panel'),
                  Line2D([], [], marker='s', ls='', ms=5.0, mfc=COL[NP], mec=COL[NP],
                         label='non-pathway-300 panel'),
                  Line2D([], [], marker='s', ls='', ms=5.0, mfc='white', mec=GREY,
                         label='$P\\geq0.05$')],
         loc='lower right', bbox_to_anchor=(1.0, 1.012), ncol=2, frameon=False,
         fontsize=7.5, handlelength=0.8, handletextpad=0.35, labelspacing=0.28,
         borderpad=0.0, columnspacing=1.6)
for ax in (A, B):
    card(fig, ax)
panel_letters(fig, [(A, 'a'), (B, 'b')])
p = os.path.join(OUT, 'FigS3_Panel.pdf')
fig.savefig(p)
fig.savefig(p.replace('.pdf', '.png'), dpi=400)
plt.close(fig)
print('[SAVED]', p, '%.1f KB' % (os.path.getsize(p) / 1024))


print('S3a panel pairs:', {t: D['panel_pairs'][t] for t in [KP, NP]})
