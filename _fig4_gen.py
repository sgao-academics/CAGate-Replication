# -*- coding: utf-8 -*-
"""Figure 4 -- consensus across cancer types, drawn in the house style
(house style).

Figure 4 carries the *cross-environment* half of the argument: an edge that
recurs in more cancer types is far more likely to be an edge that an
independent database knows about.  The four panels are:

  a) the consensus ladder -- observed external support against the uniform
     label-permutation null and against the stricter degree-matched null;
     the nulls are flat, the observation climbs, and the two cross at cnt=3
  b) sign stability -- the consensus edges do not change sign between cancers
  c) multi-source, multi-panel forest -- the effect is not one database and
     not one gene panel
  d) degree x strength matched strata -- the consensus signal survives when
     both degree and edge strength are held fixed

Chart-type discipline.  Fig.1 uses a residual strip, a logistic curve and
capsules; Fig.2 uses scatter, paired scatter and capsules; Fig.3 uses a violin,
a paired column and a slope.  This figure therefore uses four further idioms:
a ribbon ladder, stacked proportion bars, a dodged forest, and a stratified
dot strip -- and no capsules or violins.

Canvas width is the manuscript's own text block, 127.3 mm (361 pt, measured
from ws-jbcb), because the figure is placed at \\textwidth.

Every number is read from fig4_data.json, assembled by _fig4_data.py from the
round-6..8 evidence files; nothing is re-typed.
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
OUT = os.path.join(HERE, 'figures')   # figure output
os.makedirs(OUT, exist_ok=True)
W = 127.3 * MM                                   # == \textwidth of ws-jbcb
D = json.load(open(os.path.join(C, 'fig4_data.json'), encoding='utf-8'))

OBS, NULL, AUX = PAL['mist'], PAL['lilac'], PAL['violet']
SEC = PAL['peri']
PCT = FuncFormatter(lambda v, p: '%d' % round(100 * v))


def rbox(ax, x, y, s, ha='right', va='top', size=7.5):
    ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va, fontsize=size, color=INK,
            linespacing=1.35,
            bbox=dict(boxstyle='round,pad=0.34', fc=BOXFC, ec=BOXEC, lw=0.55), zorder=9)


fig = plt.figure(figsize=(W, 116.0 * MM))
gs = gridspec.GridSpec(3, 2, height_ratios=[1.0, 0.78, 0.66],
                       hspace=0.78, wspace=0.50,
                       left=0.132, right=0.986, top=0.944, bottom=0.098)
A = fig.add_subplot(gs[0, 0])
B = fig.add_subplot(gs[0, 1])
Cc = fig.add_subplot(gs[1, :])
Dd = fig.add_subplot(gs[2, :])

# ══════════ a) the consensus ladder (STRING, d=300) ══════════
LAD = D['ladder']['STRING']
x = np.arange(len(LAD))
obs = [r['obs'] for r in LAD]
uni = [r['unif'] for r in LAD]
dmn = [r['dm'] for r in LAD]
A.axvspan(-0.46, 1.46, color=TRACK, lw=0, zorder=0)             # cnt <= 3
A.plot(x, uni, ls=(0, (3.2, 2.6)), lw=1.0, color=GREY, zorder=2)
A.plot(x, dmn, '-', lw=1.4, color=NULL, zorder=3,
       marker='o', ms=3.4, mfc='white', mec=NULL, mew=0.9)
A.plot(x, obs, '-', lw=1.9, color=OBS, zorder=5, solid_capstyle='round')
for xi, yi in zip(x, obs):
    dots(A, [xi], [yi], 30, 'white', z=5.6, lw=0.0)
    dots(A, [xi], [yi], 21, OBS, z=5.8, lw=0.6)
A.set_xlim(-0.46, len(LAD) - 0.54)
A.set_ylim(-0.05, 1.14)
A.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
A.yaxis.set_major_formatter(PCT)
A.set_xticks(x)
A.set_xticklabels(['1', '2\u20133', '4\u20137', '8\u201315', '\u226516'])
A.set_xlabel('cancer types in which the edge recurs')
A.set_ylabel('STRING support (%)')
tidy(A, grid='y')
haloed_text(A, 4.40, 0.072, 'uniform null', size=7.5, color=GREY, ha='right', va='bottom')
haloed_text(A, 2.55, dmn[3] - 0.052, 'degree-matched null', size=7.5, color=NULL,
            ha='center', va='top', fontweight='bold')
haloed_text(A, 3.35, obs[3] + 0.050, 'observed', size=7.5, color=OBS,
            ha='center', va='bottom', fontweight='bold')


# ══════════ b) sign stability ══════════
SG = D['sign']['d300']
bands = ['2\u20133', '4\u20137', '8\u201315', '\u226516']
xa = np.arange(len(bands))
w = 0.34
o = [SG[b]['all_same'] for b in ['2-3', '4-7', '8-15', '16+']]
n = [SG[b]['all_same_null'] for b in ['2-3', '4-7', '8-15', '16+']]
B.bar(xa - w / 2, o, w, color=OBS, alpha=0.94, edgecolor='white', linewidth=0.4, zorder=3)
B.bar(xa + w / 2, n, w, color=NULL, alpha=0.94, edgecolor='white', linewidth=0.4, zorder=3)
for i in range(len(bands)):
    B.text(xa[i] - w / 2, o[i] + 0.030, '%.1f' % (100 * o[i]), ha='center', va='bottom',
           fontsize=7.5, color=INK)
    if n[i] > 0.04:
        B.text(xa[i] + w / 2, n[i] + 0.030, '%.0f' % (100 * n[i]), ha='center', va='bottom',
               fontsize=7.5, color=INK)
B.set_xlim(-0.60, len(bands) - 0.40)
B.set_ylim(0, 1.32)
B.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
B.yaxis.set_major_formatter(PCT)
B.set_xticks(xa)
B.set_xticklabels(bands)
B.set_xlabel('cancer types in which the edge recurs')
B.set_ylabel('single-sign edges (%)')
tidy(B, grid='y')
B.text(0.02, 0.995, 'observed', transform=B.transAxes, ha='left', va='top',
       fontsize=7.5, color=OBS, fontweight='bold')
B.text(0.32, 0.995, '50/50 null', transform=B.transAxes, ha='left', va='top',
       fontsize=7.5, color=GREY)

# ══════════ c) forest: source x panel ══════════
SRCS = ['STRING', 'BioGRID', 'Hallmark', 'KEGG']
PL = [('KEGG-300', OBS), ('KEGG-100', SEC), ('non-pathway-300', AUX)]
rows = {}
for r in D['forest']:
    rows.setdefault(r['source'], {})[r['panel']] = r
for i, s in enumerate(SRCS):
    yc = len(SRCS) - 1 - i
    Cc.axhline(yc, color=GRID, lw=0.7, zorder=0)
    for k, (p, col) in enumerate(PL):
        if p not in rows.get(s, {}):
            continue
        r = rows[s][p]
        a, b = r['a'], r['n_hi'] - r['a']
        c, d = r['c'], r['n_lo'] - r['c']
        if min(a, b, c, d) <= 0:
            continue
        se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        lo, hi = r['OR'] * np.exp(-1.96 * se), r['OR'] * np.exp(1.96 * se)
        yy = yc + [0.200, 0.0, -0.200][k]
        sig = r['p'] < 0.05
        Cc.plot([lo, hi], [yy, yy], '-', color=col, lw=0.9, alpha=0.60, zorder=3)
        Cc.plot([lo, lo], [yy - 0.055, yy + 0.055], '-', color=col, lw=0.8, alpha=0.60, zorder=3)
        Cc.plot([hi, hi], [yy - 0.055, yy + 0.055], '-', color=col, lw=0.8, alpha=0.60, zorder=3)
        dots(Cc, [r['OR']], [yy], 26, 'white', z=5.6, lw=0.0)
        dots(Cc, [r['OR']], [yy], 17, col if sig else 'white', z=5.8, lw=0.0)
        if not sig:
            Cc.plot([r['OR']], [yy], 'o', ms=3.4, mfc='white', mec=col, mew=1.0, zorder=6)
Cc.axvline(1.0, color=GREY, ls=(0, (3.0, 2.4)), lw=0.9, zorder=1)
Cc.set_xscale('log')
Cc.set_xlim(0.85, 78)
Cc.set_xticks([1, 2, 5, 10, 20, 50])
Cc.set_xticklabels(['1', '2', '5', '10', '20', '50'])
Cc.set_ylim(-0.58, 4.72)          # headroom so the legend clears the STRING row
Cc.set_yticks(range(len(SRCS)))
Cc.set_yticklabels(SRCS[::-1])
Cc.set_xlabel('Fisher odds ratio for external support (cnt $\\geq$ 4 vs $<$ 4)')
tidy(Cc, grid='x')
h = [Line2D([], [], marker='o', ls='-', ms=3.6, mfc=OBS, mec=OBS, color=OBS, lw=0.9,
            alpha=0.95, label='KEGG-300'),
     Line2D([], [], marker='o', ls='-', ms=3.6, mfc=SEC, mec=SEC, color=SEC, lw=0.9,
            alpha=0.95, label='KEGG-100'),
     Line2D([], [], marker='o', ls='-', ms=3.6, mfc=AUX, mec=AUX, color=AUX, lw=0.9,
            alpha=0.95, label='non-pathway-300'),
     Line2D([], [], marker='o', ls='none', ms=3.6, mfc='white', mec=GREY, mew=1.0,
            label='$P\\geq0.05$')]
Cc.legend(handles=h, loc='lower right', bbox_to_anchor=(1.0, 1.010), frameon=False,
          fontsize=7.5, ncol=4, handlelength=0.9, handletextpad=0.35,
          labelspacing=0.28, borderpad=0.0, columnspacing=1.5).set_zorder(10)

# ══════════ d) degree x strength matched strata ══════════
ST = ['STRING', 'BioGRID', 'Hallmark']
Dd.axvspan(0, 84, color=TRACK, lw=0, zorder=0)
for i, s in enumerate(ST):
    yc = len(ST) - 1 - i
    Dd.axhline(yc, color=GRID, lw=0.7, zorder=0)
    cells = D['strat'][s]
    xs = [c['diff'] * 100 for c in cells]
    Dd.plot(xs, [yc] * len(xs), 'o', ms=3.4, mfc='white', mec=GREY, mew=0.9,
            alpha=0.95, zorder=4)
    wm = D['wdiff']['d300'][s] * 100
    Dd.plot([min(0, wm), wm], [yc, yc], '-', color=OBS, lw=2.2, zorder=5,
            solid_capstyle='round')
    Dd.plot([wm], [yc], 'o', ms=6.2, mfc='white', mec=OBS, mew=1.5, zorder=6)
    haloed_text(Dd, wm, yc + 0.20, '$+%.0f$' % wm, size=7.5, color=OBS, ha='center',
                va='bottom', fontweight='bold')
Dd.axvline(0, color=GREY, ls=(0, (3.0, 2.4)), lw=0.9, zorder=1)
Dd.set_xlim(-28, 84)
Dd.set_ylim(-0.52, 2.86)
Dd.set_yticks(range(len(ST)))
Dd.set_yticklabels(ST[::-1])
Dd.set_xticks([-20, 0, 20, 40, 60, 80])
Dd.set_xticklabels(['\u221220', '0', '+20', '+40', '+60', '+80'])
Dd.set_xlabel('support-rate difference, cnt $\\geq$ 4 $-$ cnt $<$ 4 (percentage points)')
tidy(Dd, grid='x')

for ax in (A, B, Cc, Dd):
    card(fig, ax)
panel_letters(fig, [(A, 'a'), (B, 'b'), (Cc, 'c'), (Dd, 'd')])

p = os.path.join(OUT, 'Fig4_Consensus.pdf')
fig.savefig(p)
fig.savefig(p.replace('.pdf', '.png'), dpi=400)
plt.close(fig)
print('[SAVED]', p, '%.1f KB' % (os.path.getsize(p) / 1024))
print('a) STRING obs %s | unif %s | degmatch %s' % ([round(v, 3) for v in obs],
                                                    [round(v, 3) for v in uni],
                                                    [round(v, 3) for v in dmn]))
print('b) all-same %s  null %s' % ([round(v, 3) for v in o], [round(v, 3) for v in n]))
print('c) rows %s' % {k: sorted(v) for k, v in rows.items()})
print('d) wdiff %s' % {s: round(D['wdiff']['d300'][s] * 100, 1) for s in ST})
