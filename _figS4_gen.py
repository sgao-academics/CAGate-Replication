"""Supplementary Figure S4 -- what the consensus edges are made of, drawn in
the house style.

The twelve highest-degree genes of each consensus graph, ranked by the number of
consensus edges they carry and coloured by KEGG pathway membership (hsa04060,
cytokine-cytokine receptor interaction; hsa04110, cell cycle).  The point of the
panel is that the concentration into one module is a property of the gene panel,
not of the criterion.

Canvas width is the Supplementary Material's own text block, 160.0 mm (a
stand-alone article with margin=2.5cm), not the 127.3 mm of the ws-jbcb main
text, and it is not cropped, so the declared point sizes survive 1:1.  Ranked
module-coloured bars are an idiom not used anywhere in Fig. 1-4.

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

# ══════════════════════ Figure S5 ══════════════════════
fig = plt.figure(figsize=(W, 74.0 * MM))
gs = gridspec.GridSpec(1, 2, wspace=0.46, left=0.115, right=0.985, top=0.885, bottom=0.145)
A = fig.add_subplot(gs[0, 0])
B = fig.add_subplot(gs[0, 1])
MCOL = {'cytokine-receptor': CYT, 'cell-cycle': CCY, 'other': OTH}
TITLE = {KP: 'KEGG-300 panel (112 edges)', NP: 'non-pathway-300 panel (102 edges)'}
for ax, tag in [(A, 'd300'), (B, 'np300')]:
    v = D['hubs'][tag]
    top = [(g, k, m) for (g, k), m in zip(v['top'][:12], v['module'][:12])]
    yy = np.arange(len(top))[::-1]
    vals = [k for _, k, _ in top]
    cols = [MCOL[m] for _, _, m in top]
    ax.barh(yy, vals, 0.62, color=cols, alpha=0.94, edgecolor='white',
            linewidth=0.5, zorder=3)
    for i, (g, k, m) in enumerate(top):
        ax.text(k + 0.22, yy[i], str(k), va='center', ha='left', fontsize=7.5, color=INK)
    ax.set_ylim(-0.62, len(top) + 0.78)
    ax.set_xlim(0, max(vals) + 1.9)
    ax.set_xticks(range(0, int(max(vals)) + 2, 2))
    ax.set_yticks(yy)
    ax.set_yticklabels([g for g, _, _ in top])
    ax.set_xlabel('edges carried in the consensus graph')
    tidy(ax, grid='x')
    haloed_text(ax, max(vals) + 1.8, len(top) + 0.62, TITLE[KP if tag == 'd300' else NP],
                size=7.5, color=INK, ha='right', va='top', fontweight='bold')
leg = [Line2D([], [], marker='s', ls='', ms=5.0, mfc=MCOL['cytokine-receptor'],
              mec=MCOL['cytokine-receptor'], label='cytokine receptor'),
       Line2D([], [], marker='s', ls='', ms=5.0, mfc=MCOL['cell-cycle'],
              mec=MCOL['cell-cycle'], label='cell cycle'),
       Line2D([], [], marker='s', ls='', ms=5.0, mfc=MCOL['other'], mec=MCOL['other'],
              label='other')]
A.legend(handles=leg, loc='lower right', bbox_to_anchor=(1.0, 1.012), ncol=3,
         frameon=False, fontsize=7.5, handlelength=0.8, handletextpad=0.35,
         labelspacing=0.28, borderpad=0.0, columnspacing=1.6)
for ax in (A, B):
    card(fig, ax)
panel_letters(fig, [(A, 'a'), (B, 'b')])
p = os.path.join(OUT, 'FigS4_Modules.pdf')
fig.savefig(p)
fig.savefig(p.replace('.pdf', '.png'), dpi=400)
plt.close(fig)
print('[SAVED]', p, '%.1f KB' % (os.path.getsize(p) / 1024))
print('S5 modules d300:', list(zip([g for g, _ in D['hubs']['d300']['top'][:12]],
                                   D['hubs']['d300']['module'][:12])))
print('S5 modules np300:', list(zip([g for g, _ in D['hubs']['np300']['top'][:12]],
                                    D['hubs']['np300']['module'][:12])))
