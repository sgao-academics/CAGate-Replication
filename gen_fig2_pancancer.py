"""
CAGate Fig2 - Pan-Cancer Performance (2x2 composite).
(a) CAGate vs NOTEARS delta vs sample size scatter
(b) Top-15 cancers ranked by delta
(c) Small vs large sample comparison (violin)
(d) GOLEM + DAGMA per-cancer delta at d=100
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu

# ═══════════ NATURE-LEVEL COLOR PALETTE ═══════════
C_SMALLN  = '#D55E00'
C_LARGEN  = '#0072B2'
C_GOLEM   = '#009E73'
C_DAGMA   = '#E69F00'
C_TEXT    = '#333333'
C_GRID    = '#E0E0E0'
C_CI      = '#AAAAAA'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman'],
    'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 2, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.dpi': 400, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

def panel_label(ax, letter):
    ax.text(-0.08, 1.06, letter, transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='bottom', ha='left', color=C_TEXT)

# ═══════════ PATHS ═══════════
PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(PKG_ROOT, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ═══════════ DATA ═══════════
cancer_data = json.load(open(os.path.join(PKG_ROOT, 'cagate_delta_data.json')))
cancers = [c[0] for c in cancer_data]
n_vals  = np.array([c[1] for c in cancer_data], float)
d_vals  = np.array([c[2] for c in cancer_data], float)

small_mask = n_vals < 323
large_mask = ~small_mask
small_d = d_vals[small_mask]
large_d = d_vals[large_mask]

# --- Cross-method: GOLEM + DAGMA at d=100 ---
cross = json.load(open(os.path.join(PKG_ROOT, '_cross_results.json'), encoding='utf-8'))
from collections import defaultdict
dagma_d100 = defaultdict(list)
golem_d100 = defaultdict(list)
for key, val in cross.items():
    if '_d100_' not in key: continue
    cancer = val.get('cancer', key.split('_')[0])
    if '_dagma_' in key and val.get('delta') is not None:
        dagma_d100[cancer].append(val['delta'])
    elif '_golem_' in key and val.get('delta') is not None:
        golem_d100[cancer].append(val['delta'])

# Computed stats using real scipy
spearman_rho, spearman_p = spearmanr(n_vals, d_vals)
mw_u, mw_p = mannwhitneyu(small_d, large_d, alternative='two-sided')

# ═══════════ FIGURE ═══════════
fig = plt.figure(figsize=(8.5, 7.5))
gs = fig.add_gridspec(2, 2, hspace=0.48, wspace=0.42,
                      left=0.08, right=0.98, bottom=0.06, top=0.95)

# ── (a) Scatter: delta vs log_n ──
ax = fig.add_subplot(gs[0, 0])
ax.scatter(n_vals[small_mask], d_vals[small_mask],
           c=C_SMALLN, s=36, alpha=0.85, edgecolor='white', linewidth=0.3,
           zorder=5, label='$n<323$ (%d)' % small_mask.sum())
ax.scatter(n_vals[large_mask], d_vals[large_mask],
           c=C_LARGEN, s=36, alpha=0.85, edgecolor='white', linewidth=0.3,
           zorder=5, label='$n\\geq323$ (%d)' % large_mask.sum())

# Spearman regression line + bootstrap CI
log_n = np.log10(n_vals)
cf = np.polyfit(log_n, d_vals, 1)
xf = np.logspace(log_n.min() - 0.05, log_n.max() + 0.05, 120)
yf = np.polyval(cf, np.log10(xf))
yb = np.array([np.polyfit(
    log_n[idx := np.random.choice(len(n_vals), len(n_vals), True)],
    d_vals[idx], 1) for _ in range(800)])
yl, yh = np.percentile(
    np.array([np.polyval(p, np.log10(xf)) for p in yb]), [2.5, 97.5], axis=0)
ax.fill_between(xf, yl, yh, alpha=0.15, color=C_CI, linewidth=0)
ax.plot(xf, yf, '-', color=C_TEXT, lw=1.2, alpha=0.7)
ax.axhline(0, color=C_GRID, lw=0.5, ls=':', alpha=0.5)

# Annotation
p_str = '$P=%.0e$' % spearman_p if spearman_p < 0.001 else '$P=%.4f$' % spearman_p
ax.text(0.96, 0.94, 'Spearman $\\rho=%.3f$\n%s' % (spearman_rho, p_str),
        transform=ax.transAxes, ha='right', va='top', fontsize=7.5, color=C_TEXT,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_GRID, alpha=0.92, lw=0.4))

ax.set_xlabel('Sample size $n$')
ax.set_ylabel('$\\Delta$ (CAGate $-$ NOTEARS)')
ax.legend(fontsize=6.5, loc='lower left', framealpha=0.85,
          handletextpad=0.4, markerscale=0.8, labelspacing=0.3)
ax.set_xlim(left=20)
panel_label(ax, 'a')

# ── (b) Top-15 cancers ranked ──
ax = fig.add_subplot(gs[0, 1])
sorted_idx = np.argsort(d_vals)[::-1]
top_n = 15
top_c  = [cancers[i] for i in sorted_idx[:top_n][::-1]]
top_d  = [d_vals[i]  for i in sorted_idx[:top_n][::-1]]
top_ns = [n_vals[i]  for i in sorted_idx[:top_n][::-1]]
colors_bar = [C_SMALLN if ns < 323 else C_LARGEN for ns in top_ns]

ax.barh(range(top_n), top_d, color=colors_bar, height=0.62, alpha=0.9,
        edgecolor='white', linewidth=0.3)
for i, (d, ns) in enumerate(zip(top_d, top_ns)):
    ax.text(max(d + 12, 12), i, '%d' % int(d),
            va='center', fontsize=6.8, color=C_TEXT, weight='bold')

ax.set_yticks(range(top_n))
ax.set_yticklabels(['%s (%d)' % (c, ns) for c, ns in zip(top_c, top_ns)], fontsize=7)
ax.set_xlabel('$\\Delta$ (CAGate $-$ NOTEARS)')
ax.invert_yaxis()
ax.set_xlim(right=max(top_d) * 1.18)
panel_label(ax, 'b')

# ── (c) Small vs large sample ──
ax = fig.add_subplot(gs[1, 0])
vp = ax.violinplot([small_d, large_d], positions=[1, 2],
                    showmeans=False, showmedians=True, widths=0.55)
for pc, color in zip(vp['bodies'], [C_SMALLN, C_LARGEN]):
    pc.set_facecolor(color); pc.set_alpha(0.30); pc.set_edgecolor(color); pc.set_linewidth(0.8)
vp['cmedians'].set_color(C_TEXT); vp['cmedians'].set_linewidth(1.5)

for idx, data in enumerate([small_d, large_d]):
    jitter = np.random.default_rng(42).normal(0, 0.04, len(data))
    ax.scatter(np.full(len(data), idx + 1) + jitter, data,
               c=[C_SMALLN, C_LARGEN][idx], s=16, alpha=0.55,
               edgecolor='white', linewidth=0.2, zorder=10)

# MW annotation: top-left corner
ax.text(0.03, 0.94, 'Mann--Whitney $U=%d$\n$P=%.0e$' % (mw_u, mw_p),
        transform=ax.transAxes, ha='left', va='top',
        fontsize=7.5, color=C_TEXT,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_GRID, alpha=0.92, lw=0.4))

ax.set_xticks([1, 2])
ax.set_xticklabels(['Small\n($n<323$, %d cancers)' % small_mask.sum(),
                    'Large\n($n\\geq323$, %d cancers)' % large_mask.sum()], fontsize=7.2)
ax.set_ylabel('$\\Delta$ (CAGate $-$ NOTEARS)')
ax.set_xlim(0.3, 2.7)
panel_label(ax, 'c')

# ── (d) GOLEM + DAGMA per-cancer delta at d=100 ──
ax = fig.add_subplot(gs[1, 1])

# Use cancers that appear in BOTH methods
common = sorted(set(dagma_d100.keys()) & set(golem_d100.keys()))
dagma_m = np.array([np.mean(dagma_d100[c]) for c in common])
golem_m = np.array([np.mean(golem_d100[c]) for c in common])

# Sort by DAGMA delta (the one with variance)
order = np.argsort(dagma_m)[::-1]  # worst (most positive) at top
show_n = min(18, len(common))
order = order[-show_n:]  # take top 18 worst
show_c = [common[i] for i in order]
show_d = dagma_m[order]
show_g = golem_m[order]

y = np.arange(show_n)
h = 0.32
ax.barh(y - h/2, show_g, h, color=C_GOLEM, alpha=0.85, label='GOLEM',
        edgecolor='white', linewidth=0.2)
ax.barh(y + h/2, show_d, h, color=C_DAGMA, alpha=0.85, label='DAGMA',
        edgecolor='white', linewidth=0.2)
ax.axvline(0, color=C_TEXT, lw=0.8, alpha=0.4)

# Value labels only for DAGMA (GOLEM is too flat to label)
for i, m in enumerate(show_d):
    off = 0.4 if m >= 0 else -0.6
    ax.text(m + off, i + h/2, '%+.0f' % m, va='center', fontsize=6, color=C_DAGMA,
            ha='left' if m >= 0 else 'right', weight='bold')

ax.set_yticks(y)
ax.set_yticklabels(show_c, fontsize=7)
ax.set_xlabel('$\\Delta$ (Method $-$ CAGate) at $d=100$')
ax.invert_yaxis()

# Legend: horizontal, top-left, above annotation
ax.legend(fontsize=6.5, loc='upper left', framealpha=0.85, ncol=2)

# Summary annotation: below legend, top-left
g_mean = np.mean(golem_m)
d_mean = np.mean(dagma_m)
ax.text(0.03, 0.86, 'DAGMA mean $\\Delta=%.0f$\nGOLEM mean $\\Delta=%.1f$' % (d_mean, g_mean),
        transform=ax.transAxes, ha='left', va='top',
        fontsize=7, color=C_TEXT,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_GRID, alpha=0.92, lw=0.4))

panel_label(ax, 'd')

# ═══════════ SAVE ═══════════
out = os.path.join(FIG_DIR, 'Fig2_Pancancer.pdf')
fig.savefig(out, dpi=400, facecolor='white', edgecolor='none')
plt.close(fig)
print('[SAVED] Fig2_Pancancer.pdf (%d KB)' % (os.path.getsize(out) // 1024))
