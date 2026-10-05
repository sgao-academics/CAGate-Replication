# -*- coding: utf-8 -*-
"""Figure S1 -- high-dimensional behaviour, drawn in the house style.

Chart-type discipline: Figures 1-3 already spend a residual strip, a logistic
curve, capsules, a scatter, a paired scatter, a violin, a paired column and a
slope.  This figure adds the one idiom the set was missing -- a multi-series
trajectory against dimension -- and it is placed in the Supplementary Material,
where the argument (no collapse) is made.

(a) synthetic Erdos-Renyi sweep (n = 500): the converged solver tracks the true
    edge count at every dimension; the gate returns slightly fewer.
(b) real TCGA-BRCA (n = 1218): the same two arms against dimension.

Canvas = 160.0 mm -- the Supplementary Material's own text block (a stand-alone
article with margin=2.5cm), NOT the 127.3 mm of the ws-jbcb main text -- and it
is not cropped, so the declared point sizes survive 1:1.
Every number comes from the corrected solver's own evidence files.
"""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from figstyle import MM, PAL, INK, GREY, card, panel_letters, tidy

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

EV = json.load(open(os.path.join(C, 'evidence.json'), encoding='utf-8'))
SY = sorted(EV['synth'], key=lambda j: j['d'])
RE = [json.load(open(f, encoding='utf-8'))
      for f in sorted(glob.glob(os.path.join(C, 'real_dim', 'brca_d*.json')))]
RE.sort(key=lambda j: j['d'])

NOT, BAS, GAT = PAL['lilac'], PAL['peri'], PAL['mist']   # reference / cluster-only / gate

fig = plt.figure(figsize=(W, 78.0 * MM))
gs = gridspec.GridSpec(1, 2, wspace=0.30, left=0.098, right=0.985, top=0.862, bottom=0.158)
A = fig.add_subplot(gs[0, 0])
B = fig.add_subplot(gs[0, 1])


def series(ax, xs, y, color, marker, lw, ms, lab, z=5):
    ax.plot(xs, y, '-', color=color, lw=lw, zorder=z - 1)
    ax.plot(xs, y, marker, color=color, ms=ms, mfc=color, mec='white', mew=0.55,
            zorder=z, label=lab)


# ══════════ a) synthetic Erdos-Renyi sweep ══════════
xs = np.arange(len(SY))
tru = [j['n_true_edges'] for j in SY]
A.plot(xs, tru, '--', color=GREY, lw=1.05, zorder=1, label='true edges')
series(A, xs, [j['notears']['edges_0.3'] for j in SY], NOT, 'o', 1.15, 3.6, 'NOTEARS')
series(A, xs, [j['cagate_c_base']['edges_0.3'] for j in SY], BAS, 's', 1.15, 3.3, 'CAGate base')
series(A, xs, [j['cagate_c_gate']['edges_0.3'] for j in SY], GAT, '^', 1.30, 3.8, 'CAGate gate')
A.set_xlim(-0.44, len(SY) - 0.56)
A.set_ylim(0, 218)
A.set_yticks([0, 50, 100, 150, 200])
A.set_xticks(xs)
A.set_xticklabels([r'$\mathbf{%d}$' % j['d'] for j in SY])
A.set_xlabel('genes $d$')
A.set_ylabel('edges recovered')
A.legend(loc='upper left', frameon=False, fontsize=7.5, handlelength=1.4,
         handletextpad=0.5, labelspacing=0.30, borderpad=0.10)
tidy(A, grid='y')

# ══════════ b) real BRCA ══════════
xs = np.arange(len(RE))
series(B, xs, [j['notears']['edges'] for j in RE], NOT, 'o', 1.15, 3.6, 'NOTEARS')
series(B, xs, [j['base']['edges'] for j in RE], BAS, 's', 1.15, 3.3, 'CAGate base')
series(B, xs, [j['gate']['edges'] for j in RE], GAT, '^', 1.30, 3.8, 'CAGate gate')
for i, j in enumerate(RE):
    B.text(xs[i], j['notears']['edges'] + 3.0, '%d' % j['notears']['edges'],
           ha='center', va='bottom', fontsize=7.5, color=INK)
B.set_xlim(-0.44, len(RE) - 0.56)
B.set_ylim(0, 88)
B.set_yticks([0, 20, 40, 60, 80])
B.set_xticks(xs)
B.set_xticklabels([r'$\mathbf{%d}$' % j['d'] for j in RE])
B.set_xlabel('genes $d$')
B.set_ylabel('edges recovered')
B.legend(loc='upper left', frameon=False, fontsize=7.5, handlelength=1.4,
         handletextpad=0.5, labelspacing=0.30, borderpad=0.10)
tidy(B, grid='y')

for ax in (A, B):
    card(fig, ax)
panel_letters(fig, [(A, 'a'), (B, 'b')])

p = os.path.join(OUT, 'FigS1_HighDim.pdf')
fig.savefig(p)
fig.savefig(p.replace('.pdf', '.png'), dpi=400)
plt.close(fig)
print('[SAVED]', p, '%.1f KB' % (os.path.getsize(p) / 1024))
print('a) d      %s' % [j['d'] for j in SY])
print('   true   %s' % tru)
print('   NOT    %s' % [j['notears']['edges_0.3'] for j in SY])
print('   gate   %s' % [j['cagate_c_gate']['edges_0.3'] for j in SY])
print('b) d      %s' % [j['d'] for j in RE])
print('   NOT    %s' % [j['notears']['edges'] for j in RE])
print('   base   %s' % [j['base']['edges'] for j in RE])
print('   gate   %s' % [j['gate']['edges'] for j in RE])
