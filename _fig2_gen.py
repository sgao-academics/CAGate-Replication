# -*- coding: utf-8 -*-
"""Figure 2, drawn in the house style (作图心得与模板/00_核心规范).

Canvas width is the manuscript's own text block, 127.3 mm (361 pt, measured from
ws-jbcb), because Fig. 2 is placed at \\textwidth: drawing at the library's 174 mm
would scale every 8 pt label down to 5.8 pt and break WSPC's 8 pt floor.

The figure tells one story --- gating trades edge volume for edge precision:

  a) volume-precision trade-off: the gate gives up edges and buys external support
  b) external support (STRING) per arm, against random edges
  c) per-cancer paired gain of the gate over NOTEARS (32 of 33 above the diagonal)
  d) curated regulatory pairs (TRRUST), either direction

Every number comes from the corrected solver's own evidence files.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from figstyle import (MM, PAL, INK, GREY, TRACK, BOXFC, BOXEC, GRID,
                      card, dots, capsule, haloed_text, panel_letters, tidy)

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

EV = json.load(open(os.path.join(C, 'evidence.json'), encoding='utf-8'))
rows = EV['per_cancer']
T = EV['tests']
dv = json.load(open(os.path.join(C, 'dirval_results.json'), encoding='utf-8'))
cancers = sorted(dv)

ARMS = ['gate', 'base', 'notears']
ARM_LBL = {'gate': 'CAGate gate', 'base': 'CAGate base', 'notears': 'NOTEARS'}
ARM_C = {'gate': PAL['mist'], 'base': PAL['peri'], 'notears': PAL['moss']}
ARM_M = {'gate': 'o', 'base': '^', 'notears': 's'}
S = {'gate': 's_gate', 'base': 's_base', 'notears': 's_notears'}


def rbox(ax, x, y, s, ha='right', va='top', size=7.5):
    """Rounded annotation container, the house idiom for a stat callout."""
    ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va, fontsize=size, color=INK,
            linespacing=1.35,
            bbox=dict(boxstyle='round,pad=0.34', fc=BOXFC, ec=BOXEC, lw=0.55), zorder=9)


def capsule_summary(ax, values, y, color, h=0.30, lw=0.75):
    """IQR as a capsule on a whisker line, with a median tick -- a box, restyled."""
    v = np.asarray(values, float)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    ax.plot([v.min(), v.max()], [y, y], color=color, lw=lw, alpha=0.75, zorder=2,
            solid_capstyle='round')
    capsule(ax, float(q1), float(q3), y, h, color, z=3)
    ax.plot([med, med], [y - h * 0.72, y + h * 0.72], color='white', lw=1.5, zorder=4)
    ax.plot([med, med], [y - h * 0.72, y + h * 0.72], color=INK, lw=0.7, zorder=4.2)
    dots(ax, [float(v.mean())], [y], 26, 'white', z=6, lw=0.0, marker='o')
    dots(ax, [float(v.mean())], [y], 18, color, z=6.2, lw=0.5, marker='o')


fig = plt.figure(figsize=(W, 112.0 * MM))
gs = gridspec.GridSpec(2, 2, hspace=0.52, wspace=0.42,
                       left=0.090, right=0.982, bottom=0.090, top=0.930)
A = fig.add_subplot(gs[0, 0])
B = fig.add_subplot(gs[0, 1])
Cc = fig.add_subplot(gs[1, 0])
D = fig.add_subplot(gs[1, 1])

# ══════════ a) volume-precision trade-off ══════════
rng = np.random.default_rng(7)
A.set_xlim(16, 100)
A.set_ylim(0, 49)
for a in ARMS:
    ed = np.array([r[a] for r in rows], float)
    sp = np.array([r[S[a]] for r in rows], float)
    A.scatter(ed + rng.normal(0, 1.05, len(ed)), sp, s=7.5, marker=ARM_M[a],
              c=ARM_C[a], alpha=0.42, edgecolors='white', linewidths=0.25, zorder=4,
              label=ARM_LBL[a])
for a in ARMS:
    ed = np.array([r[a] for r in rows], float)
    sp = np.array([r[S[a]] for r in rows], float)
    mx, my = ed.mean(), sp.mean()
    A.errorbar(mx, my, xerr=ed.std(ddof=1) / np.sqrt(len(ed)),
               yerr=sp.std(ddof=1) / np.sqrt(len(sp)),
               ecolor=ARM_C[a], elinewidth=0.7, capsize=1.8, zorder=6)
    dots(A, [mx], [my], 54, ARM_C[a], z=7, lw=0.85, marker=ARM_M[a])
A.annotate('', xy=(50.3, 29.85), xytext=(63.7, 25.42),
           arrowprops=dict(arrowstyle='-|>', color=GREY, lw=0.9, shrinkA=6, shrinkB=6,
                           connectionstyle='arc3,rad=-0.22'), zorder=6)
A.set_xlabel('edges recovered')
A.set_ylabel('STRING-supported (%)')
tidy(A, grid='both')
# the legend goes ABOVE the panel: every in-axes position covers 8-27 of the
# 99 per-cancer points, whereas a one-row strip over the axes covers none
_h, _ = A.get_legend_handles_labels()
A.legend(_h, ['gate', 'base', 'NOTEARS'], loc='lower left', ncol=3,
         bbox_to_anchor=(0.005, 1.022), bbox_transform=A.transAxes, frameon=False,
         fontsize=7.5, handlelength=0.8, handletextpad=0.3, columnspacing=1.0,
         markerscale=1.5)

# ══════════ b) external support per arm ══════════
B.set_xlim(0, 46)
B.set_ylim(-0.72, 2.95)          # head band so the callout clears the gate capsule
rnd_b = float(np.mean([r['s_rand'] for r in rows]))
B.axvline(rnd_b, color=GREY, ls=(0, (2.4, 2.2)), lw=0.9, zorder=1)
B.text(rnd_b + 1.35, 0.985, 'random', transform=B.get_xaxis_transform(),
       fontsize=7.5, color=GREY, ha='left', va='top')
for i, a in enumerate(ARMS):
    capsule_summary(B, [r[S[a]] for r in rows], 2 - i, ARM_C[a])
B.set_yticks([2, 1, 0])
B.set_yticklabels(['CAGate gate', 'CAGate base', 'NOTEARS'])
B.set_xlabel('STRING-supported (%)')
tidy(B, grid='x')
rbox(B, 0.975, 0.985, 'gate $-$ NOTEARS\n$+4.43$ pp  ($P=1.6\\times10^{-8}$)')

# ══════════ c) paired per-cancer gain ══════════
Cc.set_xlim(-1.5, 36)
Cc.set_ylim(-1.5, 36)
Cc.plot([0, 36], [0, 36], color=GREY, ls=(0, (3, 2.2)), lw=0.9, zorder=1)
nx = np.array([r['s_notears'] for r in rows], float)
gy = np.array([r[S['gate']] for r in rows], float)
Cc.scatter(nx, gy, s=15, c=PAL['mist'], alpha=0.80, edgecolors='white',
           linewidths=0.45, zorder=5)
Cc.annotate('', xy=(nx.mean(), gy.mean()), xytext=(nx.mean(), nx.mean()),
            arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.1), zorder=7)
Cc.text(0.975, 0.095, 'above the diagonal:\ngate more precise\n($+4.4$ pp mean shift)',
        transform=Cc.transAxes, ha='right', va='bottom', fontsize=7.5, color=GREY,
        linespacing=1.35)
Cc.set_xlabel('NOTEARS: STRING-supported (%)')
Cc.set_ylabel('gate: STRING-supported (%)')
tidy(Cc, grid='both')
rbox(Cc, 0.055, 0.955, '$32$/$33$ above', ha='left', va='top')

# ══════════ d) curated regulatory pairs ══════════
D.set_xlim(0, 24)
D.set_ylim(-0.72, 2.95)
gr = [[dv[c][a]['either_pct'] for c in cancers] for a in ARMS]
# the paper's random baseline is each arm's OWN null over the same gene sets;
# the quoted 0.26% / 18x in the text is the gate's, so use that line here too
rnd_d = float(np.mean([dv[c]['gate']['rnd_pct'] for c in cancers]))
D.axvline(rnd_d, color=GREY, ls=(0, (2.4, 2.2)), lw=0.9, zorder=1)
D.text(rnd_d + 1.15, 0.985, 'random', transform=D.get_xaxis_transform(),
       fontsize=7.5, color=GREY, ha='left', va='top')
for i, a in enumerate(ARMS):
    capsule_summary(D, gr[i], 2 - i, ARM_C[a])
D.set_yticks([2, 1, 0])
D.set_yticklabels(['CAGate gate', 'CAGate base', 'NOTEARS'])
D.set_xlabel('In TRRUST, either direction (%)')
tidy(D, grid='x')
rbox(D, 0.975, 0.985, 'gate $%.0f\\times$ random' % (np.mean(gr[0]) / rnd_d))

for ax in (A, B, Cc, D):
    card(fig, ax)
panel_letters(fig, [(A, 'a'), (B, 'b'), (Cc, 'c'), (D, 'd')])

p = os.path.join(OUT, 'Fig2_Pancancer.pdf')
fig.savefig(p)
fig.savefig(p.replace('.pdf', '.png'), dpi=400)
plt.close(fig)
print('[SAVED]', p, '%.1f KB' % (os.path.getsize(p) / 1024))
print('means  gate %.1f/%.2f  base %.1f/%.2f  notears %.1f/%.2f' % (
    np.mean([r['gate'] for r in rows]), np.mean([r[S['gate']] for r in rows]),
    np.mean([r['base'] for r in rows]), np.mean([r[S['base']] for r in rows]),
    np.mean([r['notears'] for r in rows]), np.mean([r[S['notears']] for r in rows])))
print('trrust gate %.2f%% rnd %.2f%%  x%.1f' % (
    np.mean(gr[0]), rnd_d, np.mean(gr[0]) / rnd_d))
