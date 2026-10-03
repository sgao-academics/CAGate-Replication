# -*- coding: utf-8 -*-
"""
NEW Fig. 3 (main text) — mechanism, scope, and the precision/recall trade-off.

Replaces the old "external validation" figure whose ten per-cancer CTD / ClinGen
/ STRING constants were hard-coded and whose provenance could not be recovered.

Panels
  (a) Synthetic data WITH real cluster heterogeneity: per-seed Delta F1 (gate-base)
      for three configurations -> the gate helps (27/30 seeds).
  (b) Synthetic Erdos-Renyi data WITHOUT cluster structure: gate-base Delta SHD
      -> the gate does not help; this delimits the scope of the claim.
  (c) Recall vs precision: edges per cancer (x) against STRING edge-level support
      (y) for all three arms -> the gate sits up and to the left.
  (d) Enrichment over random edges (STRING and TRRUST), three arms.

Canvas: 5.0 in wide (= JBCB text width) so 8 pt labels print at 8 pt.
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
OUT = os.path.join(HERE, 'figures')
os.makedirs(OUT, exist_ok=True)

C_BASE, C_GATE, C_NOT, C_RAND = '#0072B2', '#D55E00', '#009E73', '#999999'
C_TEXT, C_GRID = '#333333', '#E0E0E0'
ARM_C = {'base': C_BASE, 'gate': C_GATE, 'notears': C_NOT}
ARMS = ['base', 'gate', 'notears']
ARM_LBL = {'base': 'CAGate base', 'gate': 'CAGate gate', 'notears': 'NOTEARS'}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7.5, 'axes.labelsize': 8,
    'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8, 'legend.fontsize': 6.4,
    'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 2, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
})

EV = json.load(open(os.path.join(C, 'evidence.json'), encoding='utf-8'))
rows = EV['per_cancer']
dv = json.load(open(os.path.join(C, 'dirval_results.json'), encoding='utf-8'))
cancers = sorted(dv)

cl = EV['cluster']
syn = {j['d']: j for j in EV['synth']}


def plabel(ax, s):
    ax.text(-0.19, 1.05, s, transform=ax.transAxes, fontsize=9.5,
            fontweight='bold', va='bottom', ha='left', color=C_TEXT)


def box(ax, groups, colors, labels, ylabel, seed=0, width=0.5):
    rng = np.random.default_rng(seed)
    bp = ax.boxplot(groups, positions=range(1, len(groups) + 1), widths=width,
                    showfliers=False, patch_artist=True,
                    medianprops=dict(color=C_TEXT, lw=1.1),
                    boxprops=dict(lw=0.5), whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5))
    for p, c in zip(bp['boxes'], colors):
        p.set_facecolor(c); p.set_alpha(0.16); p.set_edgecolor(c)
    for i, (g, c) in enumerate(zip(groups, colors), start=1):
        j = rng.normal(0, 0.055, len(g))
        ax.scatter(np.full(len(g), i) + j, g, s=7, c=c, alpha=0.75,
                   edgecolor='white', linewidth=0.2, zorder=5)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)


fig = plt.figure(figsize=(5.0, 4.45))
gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.50,
                      left=0.11, right=0.98, bottom=0.10, top=0.955)

# ── (a) per-seed Delta F1 on cluster-structured synthetic data ──
ax = fig.add_subplot(gs[0, 0])
labels = ['$d$15\n$K$5', '$d$15\n$K$8', '$d$20\n$K$6']
deltas = []
for j in cl:
    b = np.array(j['base_all']); g = np.array(j['gate_all'])
    deltas.append(g - b)
box(ax, deltas, [C_GATE] * 3, labels, '$\\Delta$F1 (gate $-$ base)', seed=1, width=0.55)
ax.axhline(0, color=C_TEXT, lw=0.7, alpha=0.5)
wins = sum(int((d > 0).sum()) for d in deltas)
tot = sum(len(d) for d in deltas)
ax.text(0.97, 0.04, '%d / %d seeds positive' % (wins, tot),
        transform=ax.transAxes, ha='right', va='bottom', fontsize=6.3, color=C_TEXT,
        bbox=dict(boxstyle='round,pad=0.22', fc='white', ec=C_GRID, lw=0.4, alpha=0.95))
plabel(ax, 'a')

# ── (b) scope: homogeneous Erdos-Renyi sweep, gate-base Delta SHD ──
ax = fig.add_subplot(gs[0, 1])
ds = sorted(syn)
y = [syn[d]['cagate_c_gate']['shd'] - syn[d]['cagate_c_base']['shd'] for d in ds]
ax.bar(range(len(ds)), y, color=C_GATE, width=0.55, alpha=0.85,
       edgecolor='white', linewidth=0.3)
for i, v in enumerate(y):
    ax.text(i, v + 0.7, '%+d' % v, ha='center', va='bottom', fontsize=6.5, color=C_TEXT)
ax.axhline(0, color=C_TEXT, lw=0.7, alpha=0.5)
ax.set_xticks(range(len(ds)))
ax.set_xticklabels(['$d$%d' % d for d in ds])
ax.set_ylabel('$\\Delta$SHD (gate $-$ base)')
ax.set_ylim(min(0, min(y) - 3), max(y) + 5)
ax.text(0.97, 0.96, 'no cluster\nstructure', transform=ax.transAxes,
        ha='right', va='top', fontsize=6.3, color=C_TEXT,
        bbox=dict(boxstyle='round,pad=0.22', fc='white', ec=C_GRID, lw=0.4, alpha=0.95))
plabel(ax, 'b')

# ── (c) recall vs precision (edges vs STRING support), per cancer ──
ax = fig.add_subplot(gs[1, 0])
M = {'base': ('base', 's_base'), 'gate': ('gate', 's_gate'), 'notears': ('notears', 's_notears')}
for a in ARMS:
    ek, sk = M[a]
    ax.scatter([r[ek] for r in rows], [r[sk] for r in rows], s=8, c=ARM_C[a],
               alpha=0.55, edgecolor='white', linewidth=0.15, label=ARM_LBL[a], zorder=4)
txt = []
for a in ARMS:
    ek, sk = M[a]
    xm = np.mean([r[ek] for r in rows]); ym = np.mean([r[sk] for r in rows])
    ax.scatter([xm], [ym], s=52, marker='D', c=ARM_C[a],
               edgecolor='white', linewidth=0.7, zorder=9)
    txt.append('%s  %.1f edges, %.1f%%' % (ARM_LBL[a].replace('CAGate ', ''), xm, ym))
ax.set_xlabel('Edges per cancer')
ax.set_ylabel('STRING support (%)')
ax.set_ylim(-2, 58)
ax.legend(fontsize=6.0, loc='upper left', framealpha=0.9, handletextpad=0.3,
          borderpad=0.3, labelspacing=0.25, markerscale=1.6)
ax.text(0.97, 0.04, '\n'.join(txt), transform=ax.transAxes, ha='right', va='bottom',
        fontsize=6.0, color=C_TEXT, linespacing=1.35,
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=C_GRID, lw=0.4, alpha=0.95))
plabel(ax, 'c')

# ── (d) enrichment over random edges ──
ax = fig.add_subplot(gs[1, 1])
en_s = [EV['means']['s_' + a] / EV['means']['s_rand'] for a in ARMS]
en_t = [np.mean([dv[c][a]['either_pct'] for c in cancers]) /
        np.mean([dv[c][a]['rnd_pct'] for c in cancers]) for a in ARMS]
x = np.arange(3); w = 0.34
ax.bar(x - w / 2, en_s, w, color=C_NOT, alpha=0.85, label='STRING (PPI)',
       edgecolor='white', linewidth=0.3)
ax.bar(x + w / 2, en_t, w, color=C_GATE, alpha=0.85, label='TRRUST (curated pairs)',
       edgecolor='white', linewidth=0.3)
for i in range(3):
    ax.text(i - w / 2, en_s[i] + 0.35, '%.1f' % en_s[i], ha='center', fontsize=6.2,
            color=C_TEXT, weight='bold')
    ax.text(i + w / 2, en_t[i] + 0.35, '%.1f' % en_t[i], ha='center', fontsize=6.2,
            color=C_TEXT, weight='bold')
ax.axhline(1, ls='--', lw=0.8, color=C_RAND)
from matplotlib.lines import Line2D
h, lb = ax.get_legend_handles_labels()
h.append(Line2D([0], [0], ls='--', lw=0.8, color=C_RAND))
lb.append('random')
ax.set_xticks(x)
ax.set_xticklabels([ARM_LBL[a] for a in ARMS])
ax.set_ylabel('Enrichment ($\\times$ random)')
ax.set_ylim(0, 27)
ax.legend(h, lb, fontsize=5.8, loc='upper left', framealpha=0.92, handletextpad=0.35,
          borderpad=0.3, labelspacing=0.22)
plabel(ax, 'd')

p = os.path.join(OUT, 'Fig3_Validation.pdf')
fig.savefig(p, facecolor='white', edgecolor='none')
fig.savefig(p.replace('.pdf', '.png'), dpi=400, facecolor='white', edgecolor='none')
plt.close(fig)
print('[SAVED]', p, '%.1f KB' % (os.path.getsize(p) / 1024))
