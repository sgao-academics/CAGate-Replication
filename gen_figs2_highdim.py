# -*- coding: utf-8 -*-
"""Supplementary Fig. S3 — dimensionality, honestly.

Replaces the old panel (a) that showed NOTEARS "collapsing to zero beyond
d=150" together with cross-method panels (b)-(d) built on self-written
GOLEM/DAGMA stand-ins.

(a) synthetic Erdos-Renyi sweep (n=500): edges vs d, plus the true edge count
(b) real TCGA-BRCA (n=1218): edges vs d
"""
import os, sys, json, glob, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(HERE, 'data')
OUT = os.path.join(HERE, 'figures_supplementary')
os.makedirs(OUT, exist_ok=True)

C_BASE, C_GATE, C_NOT = '#0072B2', '#D55E00', '#009E73'
C_TEXT, C_GRID = '#333333', '#E0E0E0'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 2, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
})

EV = json.load(open(os.path.join(C, 'evidence.json'), encoding='utf-8'))
syn = sorted(EV['synth'], key=lambda j: j['d'])
real = []
for f in sorted(glob.glob(os.path.join(C, 'real_dim', 'brca_d*.json'))):
    real.append(json.load(open(f, encoding='utf-8')))
real.sort(key=lambda j: j['d'])


def plabel(ax, s):
    ax.text(-0.17, 1.05, s, transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='bottom', ha='left', color=C_TEXT)


fig = plt.figure(figsize=(5.0, 2.35))
gs = fig.add_gridspec(1, 2, wspace=0.34, left=0.105, right=0.98, bottom=0.155, top=0.90)

# ── (a) synthetic sweep ──
ax = fig.add_subplot(gs[0, 0])
ds = [j['d'] for j in syn]
ax.plot(ds, [j['n_true_edges'] for j in syn], '--', color=C_TEXT, lw=1.0,
        alpha=0.45, label='true edges', zorder=1)
ax.plot(ds, [j['notears']['edges_0.3'] for j in syn], '-o', color=C_NOT, lw=1.3,
        ms=3.4, label='NOTEARS', zorder=5)
ax.plot(ds, [j['cagate_c_base']['edges_0.3'] for j in syn], '-s', color=C_BASE,
        lw=1.3, ms=3.4, label='CAGate base', zorder=5)
ax.plot(ds, [j['cagate_c_gate']['edges_0.3'] for j in syn], '-^', color=C_GATE,
        lw=1.3, ms=3.4, label='CAGate gate', zorder=5)
ax.set_xlabel('Genes $d$')
ax.set_ylabel('Edges recovered')
ax.set_xticks(ds)
ax.set_ylim(0, 215)
ax.legend(fontsize=6.4, loc='upper left', framealpha=0.9, handletextpad=0.4,
          labelspacing=0.28, borderpad=0.3)
plabel(ax, 'a')

# ── (b) real BRCA ──
ax = fig.add_subplot(gs[0, 1])
ds = [j['d'] for j in real]
ax.plot(ds, [j['notears']['edges'] for j in real], '-o', color=C_NOT, lw=1.3,
        ms=3.4, label='NOTEARS')
ax.plot(ds, [j['base']['edges'] for j in real], '-s', color=C_BASE, lw=1.3,
        ms=3.4, label='CAGate base')
ax.plot(ds, [j['gate']['edges'] for j in real], '-^', color=C_GATE, lw=1.3,
        ms=3.4, label='CAGate gate')
for j in real:
    ax.annotate('%d' % j['notears']['edges'], (j['d'], j['notears']['edges']),
                xytext=(0, 5), textcoords='offset points', ha='center',
                fontsize=6.0, color=C_NOT)
ax.set_xlabel('Genes $d$')
ax.set_ylabel('Edges recovered')
ax.set_xticks(ds)
ax.set_ylim(0, 88)
ax.legend(fontsize=6.4, loc='upper left', framealpha=0.9, handletextpad=0.4,
          labelspacing=0.28, borderpad=0.3)
plabel(ax, 'b')

p = os.path.join(OUT, 'FigS3_HighDim.pdf')
fig.savefig(p, facecolor='white', edgecolor='none')
fig.savefig(p.replace('.pdf', '.png'), dpi=400, facecolor='white', edgecolor='none')
plt.close(fig)
print('[SAVED]', p, '%.1f KB' % (os.path.getsize(p) / 1024))
