"""FigS4: High-dimensional CAGate vs NOTEARS on synthetic Erdos-Renyi DAGs (n=500).
(a) Causal edges vs dimensionality d; (b) CAGate advantage (delta) vs d.
Data: 10-seed mean +/- SD, same as main paper Fig3.
"""
import os
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.unicode_minus'] = False

FIG_DIR = os.path.join(PKG_ROOT, 'figures_supplementary')
os.makedirs(FIG_DIR, exist_ok=True)

N_RED, N_BLUE, N_GRAY, N_DARK = '#E64B35', '#3C5488', '#95A5A6', '#2C3E50'

# ── Synthetic benchmark data (canonical, same as Fig3) ──
synth_d = np.array([10, 20, 30, 50, 75, 100, 150, 200])
cag_e = np.array([27, 54, 80, 136, 218, 312, 218, 552])
nte_e = np.array([28, 55, 82, 140, 215, 308, 12, 0])
cag_s = np.array([3, 5, 6, 11, 16, 21, 18, 34])
nte_s = np.array([3, 5, 7, 11, 17, 22, 2, 0])
delta_e = cag_e - nte_e
delta_s = np.sqrt(cag_s**2 + nte_s**2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ══════════════════════════════════════════
# (a) Edges vs dimensionality
# ══════════════════════════════════════════
ax1.errorbar(synth_d, cag_e, yerr=cag_s, fmt='o-', color=N_RED, lw=2, markersize=6,
             capsize=0, elinewidth=1, zorder=5, label='CAGate',
             markeredgecolor='white', markeredgewidth=0.3)
ax1.errorbar(synth_d, nte_e, yerr=nte_s, fmt='s--', color=N_GRAY, lw=1.5, markersize=5,
             capsize=0, elinewidth=0.8, zorder=4, label='NOTEARS',
             markeredgecolor='white', markeredgewidth=0.3)

# Failure zone — text directly inside the red shaded region
ax1.axvspan(150, 205, alpha=0.12, color=N_RED, linewidth=0, zorder=0)
ax1.text(177, 180, 'NOTEARS\nFailure\nZone', fontsize=7, color=N_RED, weight='bold',
         ha='center', va='center', alpha=0.85)

# Annotate d=200 values
ax1.annotate('552', xy=(200, 552), xytext=(185, 582),
             fontsize=7, weight='bold', color=N_RED, ha='center',
             arrowprops=dict(arrowstyle='->', color=N_RED, lw=0.6, shrinkA=0, shrinkB=1))
ax1.annotate('0', xy=(200, 0), xytext=(185, 28),
             fontsize=7, weight='bold', color=N_GRAY, ha='center')

ax1.set_xlabel(r'Dimensionality $d$')
ax1.set_ylabel('Causal edges recovered')
ax1.set_title(r'(a) Edges vs $d$ ($n=500$, Erdos--Renyi)', fontsize=11, fontweight='bold')
ax1.set_ylim(bottom=-10, top=620)
ax1.legend(fontsize=8, loc='upper left', framealpha=0.85)
ax1.text(0.02, 1.02, 'a', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='bottom')

# ══════════════════════════════════════════
# (b) CAGate advantage (delta) vs d
# ══════════════════════════════════════════
ax2.errorbar(synth_d, delta_e, yerr=delta_s, fmt='o-', color=N_BLUE, lw=2, markersize=7,
             capsize=0, elinewidth=1.2, zorder=5, markeredgecolor='white', markeredgewidth=0.3)
ax2.fill_between(synth_d, delta_e - delta_s, delta_e + delta_s,
                 alpha=0.12, color=N_BLUE, linewidth=0)
ax2.axhline(y=0, color=N_GRAY, lw=0.5, ls=':', alpha=0.5)

# Low-d cluster (d=10..100, delta -4 to +4) — compact info box in empty top-left
lo_text = '\n'.join(f'$d$={d:>3}: $\\Delta={v:+.0f}$' for d, v in zip(synth_d[:6], delta_e[:6]))
ax2.text(0.04, 0.92, lo_text, transform=ax2.transAxes,
         fontsize=6.5, color=N_DARK, va='top', ha='left',
         bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=N_GRAY, alpha=0.92, lw=0.5))
# Bracket pointing to cluster
ax2.annotate('', xy=(50, 5), xytext=(40, 50),
             arrowprops=dict(arrowstyle='->', color=N_GRAY, lw=0.5, alpha=0.5))

# High-d annotations
for d, v in [(150, 206), (200, 552)]:
    ax2.annotate(f'{v:+.0f}', xy=(d, v), xytext=(d - 10, v + 35),
                 fontsize=7.5, weight='bold', color=N_BLUE, ha='center',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#cccccc', alpha=0.9))

# Max box — right side, between curve and d=200 annotation
ax2.text(0.88, 0.82, f'Max: +{max(delta_e):.0f} at $d=200$',
         transform=ax2.transAxes, ha='right', va='bottom', fontsize=8,
         fontweight='bold', color=N_RED,
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#dddddd', alpha=0.85))

ax2.set_xlabel(r'Dimensionality $d$')
ax2.set_ylabel(r'$\Delta$ (CAGate $-$ NOTEARS, edges)')
ax2.set_title(r'(b) CAGate Advantage $\Delta$ vs $d$ ($n=500$)', fontsize=11, fontweight='bold')
ax2.text(0.02, 1.02, 'b', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'FigS4_HighDimGain.png'), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'FigS4_HighDimGain.pdf'), bbox_inches='tight')
plt.close()
print('FigS4 saved')
