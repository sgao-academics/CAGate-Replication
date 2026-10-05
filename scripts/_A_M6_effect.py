# -*- coding: utf-8 -*-
"""A-M6: effect size of the gate in absolute supported-edge terms.

The manuscript reports the precision gain (+4.43 pp) but not what it costs.
This script reads the per-cohort edge counts and STRING support rates that make
up Table S5 (shipped as data/table_s5.json) and reports the paired change in the
*number* of supported edges per cohort -- the quantity a reader needs in order to
judge whether the volume-for-precision exchange is worth making.  It also states
the multiplicity family behind the main-text paired tests.
"""
import os, sys, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from scipy.stats import wilcoxon, ttest_rel

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_HERE), 'data')
OUT = os.path.join(DATA_DIR, 'A_M6_effect.json')

D = json.load(open(os.path.join(DATA_DIR, 'table_s5.json'), encoding='utf-8'))
rows = D['cohorts']
print('cohorts %d' % len(rows))

edges = {a: np.array([r[a] for r in rows], float) for a in ('notears', 'base', 'gate')}
rate = {a: np.array([r['s_' + a] for r in rows], float) for a in ('notears', 'base', 'gate')}

res = {}
for a in ('notears', 'base', 'gate'):
    sup = edges[a] * rate[a] / 100.0
    res[a] = dict(edges=float(edges[a].mean()), rate=float(rate[a].mean()),
                  supported=float(sup.mean()))
    print('  %-8s mean edges %.2f  rate %.2f%%  implied supported/cohort %.2f'
          % (a, edges[a].mean(), rate[a].mean(), sup.mean()))

g = edges['gate'] * rate['gate'] / 100.0
nt = edges['notears'] * rate['notears'] / 100.0
ba = edges['base'] * rate['base'] / 100.0
d = g - nt
st, pv = wilcoxon(d)
tt, pt = ttest_rel(g, nt)
print('\ngate minus NOTEARS, supported edges per cohort: mean %+.2f (median %+.2f)'
      % (d.mean(), np.median(d)))
print('  Wilcoxon V=%.1f P=%.3g | paired t=%.2f P=%.3g' % (st, pv, tt, pt))
print('  cohorts where the gate has more supported edges: %d/%d' % ((d > 0).sum(), len(d)))
print('  mean edges gate %.2f vs NOTEARS %.2f -> %+.2f (%+.1f%%)'
      % (res['gate']['edges'], res['notears']['edges'],
         res['gate']['edges'] - res['notears']['edges'],
         100 * (res['gate']['edges'] / res['notears']['edges'] - 1)))
db = g - ba
print('gate minus base, supported edges per cohort: mean %+.2f (median %+.2f)'
      % (db.mean(), np.median(db)))

json.dump(dict(n=len(rows), arms=res,
               gate_vs_notears_supported=dict(mean=float(d.mean()), median=float(np.median(d)),
                                              p=float(pv), n_win=int((d > 0).sum())),
               gate_vs_base_supported=dict(mean=float(db.mean()), median=float(np.median(db))),
               per_cancer=[dict(cancer=r['cancer'], gate=round(float(g[k]), 2),
                                notears=round(float(nt[k]), 2)) for k, r in enumerate(rows)]),
          open(OUT, 'w', encoding='utf-8'), indent=1)
print('\nsaved -> %s' % OUT)
