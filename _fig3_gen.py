# -*- coding: utf-8 -*-
"""Figure 3, drawn in the house style (作图心得与模板/00_核心规范).

Figure 3 carries the *mechanism and its operating regime*, after the
volume/precision evidence moved to Figure 2:

  a) the gate raises F1 wherever there is genuine cluster heterogeneity
     (27 of 30 seeds) -- violin + per-seed dots
  b) what the gate does to the graph: across the 33 TCGA cohorts the gate
     removes well over half of the weakest |W| stratum and only a sixth of the
     strongest one, so the edges it keeps are the well-supported ones --
     a cumulative |W| profile, one line per arm
  c) the default needs no tuning: the edge count is flat across a twenty-fold
     range of the gate sensitivity alpha

Chart-type discipline: Fig.1 uses a residual strip, a logistic curve and
capsules; Fig.2 uses scatter, paired scatter and capsules; Fig.4 uses a ribbon
ladder, stacked proportion bars, a dodged forest and a stratified dot strip.
This figure therefore uses a violin, a shaded cumulative profile and a flat
response curve -- and no capsules.

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
from matplotlib.lines import Line2D
from figstyle import (MM, PAL, INK, GREY, BOXFC, BOXEC, GRID,
                      card, dots, haloed_text, panel_letters, tidy)

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
})

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(HERE, 'data')          # all inputs live here
OUT = os.path.join(HERE, 'figures')     # figure output
os.makedirs(OUT, exist_ok=True)
W = 127.3 * MM                          # == \textwidth of ws-jbcb

GATE, BASE = PAL['mist'], PAL['peri']   # same job colours as Figs. 1-2

EV = json.load(open(os.path.join(C, 'evidence.json'), encoding='utf-8'))
CL = EV['cluster']
AL = json.load(open(os.path.join(C, 'alpha_sweep.json'), encoding='utf-8'))


def rbox(ax, x, y, s, ha='right', va='top', size=7.5):
    """Rounded annotation container -- the house idiom for a stat callout."""
    ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va, fontsize=size, color=INK,
            linespacing=1.35,
            bbox=dict(boxstyle='round,pad=0.34', fc=BOXFC, ec=BOXEC, lw=0.55), zorder=9)


fig = plt.figure(figsize=(W, 84.0 * MM))
gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 0.62], hspace=0.62, wspace=0.40,
                       left=0.118, right=0.984, top=0.930, bottom=0.120)
A = fig.add_subplot(gs[0, 0])
B = fig.add_subplot(gs[0, 1])
Cc = fig.add_subplot(gs[1, :])

# ══════════ a) clustered synthetic: the gate raises F1 ══════════
deltas = [np.array(j['gate_all'], float) - np.array(j['base_all'], float) for j in CL]
lbls = ['$d$15\n$K$5', '$d$15\n$K$8', '$d$20\n$K$6']
pos = [0, 1, 2]
vp = A.violinplot(deltas, positions=pos, widths=0.74, showextrema=False,
                  showmedians=False)
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
A.set_ylim(-0.062, 0.238)
A.set_yticks([0.0, 0.05, 0.10, 0.15, 0.20])
A.set_xticks(pos)
A.set_xticklabels(lbls)
A.set_ylabel('$\\Delta$F1 (gate $-$ base)')
tidy(A, grid='y')
wins = sum(int((d > 0).sum()) for d in deltas)
tot = sum(len(d) for d in deltas)
rbox(A, 0.975, 0.985, '%d / %d seeds\nabove zero' % (wins, tot))

# ══════════ b) what the gate removes: a cumulative |W| profile ══════════
THKEY = ['1e-06', '0.001', '0.01', '0.05', '0.1', '0.3']
THVAL = np.array([1e-6, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1])
Bmat, Gmat = [], []
for f in sorted(glob.glob(os.path.join(C, 'mega33_results', '*.json'))):
    j = json.load(open(f, encoding='utf-8'))
    if 'counts' not in j.get('base', {}) or 'counts' not in j.get('gate', {}):
        continue
    Bmat.append([j['base']['counts'][k] for k in THKEY])
    Gmat.append([j['gate']['counts'][k] for k in THKEY])
Bmat, Gmat = np.array(Bmat, float), np.array(Gmat, float)
mb, mg = Bmat.mean(0), Gmat.mean(0)

xv = np.arange(len(THVAL))            # categorical axis: thresholds are decades apart
B.fill_between(xv, mg, mb, color=GATE, alpha=0.10, lw=0, zorder=1)
B.plot(xv, mb, '-o', color=BASE, lw=1.6, ms=3.4, mfc='white', mec=BASE,
       mew=0.9, zorder=4, label='ungated base')
B.plot(xv, mg, '-o', color=GATE, lw=1.6, ms=3.4, mfc='white', mec=GATE,
       mew=0.9, zorder=5, label='with gate')
B.set_yscale('log')
B.set_xlim(-0.35, len(xv) - 0.65)
B.set_ylim(30, 3600)
B.set_xlabel('$|W|$ threshold')
B.set_ylabel('edges kept per cancer (mean)')
B.set_yticks([50, 100, 500, 1000, 2000])
B.set_yticklabels(['50', '100', '500', '1000', '2000'])
B.set_xticks(xv)
B.set_xticklabels(['$10^{-6}$', '$10^{-3}$', '$10^{-2}$', '0.05', '0.1', '0.3'])
B.legend(loc='lower left', frameon=False, fontsize=7.5, handlelength=1.0,
         handletextpad=0.4, labelspacing=0.30, borderpad=0.15)
tidy(B, grid='y')
# retention ratio per end -- the mechanism, stated as a number
r_lo, r_hi = mg[0] / mb[0], mg[-1] / mb[-1]
rbox(B, 0.975, 0.985, 'gate keeps %d%% of the\nweakest stratum, %d%% of the\nstrongest'
     % (round(100 * r_lo), round(100 * r_hi)))

# ══════════ c) the default needs no tuning ══════════
al_alpha = [r['alpha'] for r in AL['rows']]
al_edges = [r['edges'] for r in AL['rows']]
base_e = AL['base']['edges']
xa = np.arange(len(al_alpha))
Cc.axhline(base_e, color=PAL['lilac'], ls=(0, (3.2, 2.6)), lw=1.1, zorder=2)
haloed_text(Cc, 0.05, base_e + 0.7, 'ungated base (%d)' % base_e,
            size=7.5, color=GREY, ha='left', va='bottom')
Cc.plot(xa, al_edges, '-', color=GATE, lw=1.9, zorder=5, solid_capstyle='round')
for xi, yi in zip(xa, al_edges):
    dots(Cc, [xi], [yi], 30, 'white', z=6.5, lw=0.0)
    dots(Cc, [xi], [yi], 21, GATE, z=6.7, lw=0.7)
Cc.set_xlim(-0.42, len(al_alpha) - 0.58)
Cc.set_ylim(46, 70)
Cc.set_yticks([50, 55, 60, 65, 70])
Cc.set_xticks(xa)
Cc.set_xticklabels(['%g' % a for a in al_alpha])
Cc.set_xlabel('gate sensitivity $\\alpha$')
Cc.set_ylabel('edges recovered\n(TCGA-BRCA, $d$200)')
tidy(Cc, grid='y')
rbox(Cc, 0.975, 0.115, '%d\u2013%d edges across a\n20-fold range of $\\alpha$'
     % (min(al_edges), max(al_edges)), ha='right', va='bottom')

for ax in (A, B, Cc):
    card(fig, ax)
panel_letters(fig, [(A, 'a'), (B, 'b'), (Cc, 'c')])

p = os.path.join(OUT, 'Fig3_Validation.pdf')
fig.savefig(p)
fig.savefig(p.replace('.pdf', '.png'), dpi=400)
plt.close(fig)
print('[SAVED]', p, '%.1f KB' % (os.path.getsize(p) / 1024))
print('a) wins %d/%d  delta means %s' % (wins, tot, [round(float(d.mean()), 4) for d in deltas]))
print('b) mean base %s' % np.round(mb, 1).tolist())
print('b) mean gate %s' % np.round(mg, 1).tolist())
print('b) retention lo/hi = %.3f / %.3f' % (r_lo, r_hi))
print('c) alpha %s edges %s base %d' % (al_alpha, al_edges, base_e))
