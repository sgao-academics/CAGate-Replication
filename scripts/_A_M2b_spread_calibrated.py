# -*- coding: utf-8 -*-
"""A-M2b: calibrate the clustered benchmark's residual-noise SPREAD to the
values measured on real transcriptomes, and re-run the gate/base comparison at
each.

Main-text clustered benchmark (SM3; the sweep archived in data/cluster_sweep/):
K clusters whose residual-noise scale c_k rises linearly across a fixed span of
2.0 (0.3 -> 2.3).  Its relative dispersion span (max - min) / median = 2.0/1.3
= 1.54, the same for all three published configurations.  The per-group
residual dispersion measured on real TCGA cohorts (Note SN25, the quantity
sigma_k the gate reads) is far narrower: 0.74 (BRCA, K-means), 0.39 (GBM),
0.20 (LUAD).

Objection (A-M2): the synthetic gain may be an artefact of an inflated span the
real data never shows.  This script re-runs the arms at the real spans, holding
the median scale fixed at 1.3 so ONLY the span changes, and includes the
published span (1.54) as a self-check: it must reproduce the archived
per-configuration deltas (data/cluster_sweep/*.json).

The published configurations and their archived deltas:
  d=15 K=5 sp=100 l1=0.01  : delta +0.0371  (9/10 seeds)
  d=15 K=8 sp=60  l1=0.01  : delta +0.0554  (9/10 seeds)
  d=20 K=6 sp=80  l1=0.005 : delta +0.0393  (9/10 seeds)

Run:  python scripts/_A_M2b_spread_calibrated.py
Writes data/spread_calib/spread_calib.json plus one checkpoint per unit.
"""
import os
import sys
import json

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (REPO_ROOT, SCRIPTS_DIR):
    sys.path.insert(0, _p)

from cagate import solve_cagate_lbfgsb, f1                       # noqa: E402

ALPHA, MEDIAN = 0.5, 1.3
CONFIGS = ((15, 5, 100, 0.01), (15, 8, 60, 0.01), (20, 6, 80, 0.005))
RELS = (('published', 1.538), ('brca', 0.74), ('gbm', 0.39), ('luad', 0.20))
SEEDS = list(range(10))
OUT = os.path.join(REPO_ROOT, 'data', 'spread_calib')
os.makedirs(OUT, exist_ok=True)


def gen_cluster_spread(d, nc, sp, seed, rel):
    """Same generator as cagate.gen_cluster, but the cluster noise scale spans
    rel * MEDIAN around a fixed median.  rel = 1.538 reproduces the published
    generator exactly (0.3 -> 2.3), so it doubles as a self-check."""
    span = rel * MEDIAN
    rng = np.random.RandomState(seed)
    Wt = np.zeros((d, d))
    e = 0
    while e < 2 * d:
        i, j = rng.randint(0, d, 2)
        if i < j and Wt[i, j] == 0:
            Wt[i, j] = rng.uniform(0.5, 2.0) * rng.choice([-1, 1])
            e += 1
    perm = rng.permutation(d)
    Wt_dag = Wt[perm][:, perm]
    inv_M = np.linalg.inv(np.eye(d) - Wt_dag)
    lo = MEDIAN - span / 2.0
    X = np.zeros((nc * sp, d))
    cid = np.zeros(nc * sp, dtype=int)
    for c in range(nc):
        cn = lo + span * c / (nc - 1)
        idx = slice(c * sp, (c + 1) * sp)
        Xc = rng.randn(sp, d) @ inv_M
        Xc += cn * rng.randn(sp, d)
        X[idx] = Xc
        cid[idx] = c
    return X, Wt_dag, cid


def unit(d, nc, sp, l1, seed, tn, rel):
    fp = os.path.join(OUT, 'd%d_k%d_sp%d_l1%g_%s_s%d.json' % (d, nc, sp, l1, tn, seed))
    if os.path.exists(fp):
        try:
            return json.load(open(fp, encoding='utf-8'))
        except Exception:
            pass
    X, Wt, cid = gen_cluster_spread(d, nc, sp, seed, rel)
    base = solve_cagate_lbfgsb(X, cid=cid, use_gate=False, l1=l1)
    gate = solve_cagate_lbfgsb(X, cid=cid, use_gate=True, gate_alpha=ALPHA, l1=l1)
    fb = f1(base['W'], Wt)[0]
    fg = f1(gate['W'], Wt)[0]
    rec = dict(d=d, nc=nc, sp=sp, l1=l1, seed=seed, rel=tn, rel_value=rel,
               f1_base=fb, f1_gate=fg, d_f1=fg - fb,
               e_base=base['n_edges'], e_gate=gate['n_edges'], h_gate=float(gate['h']))
    json.dump(rec, open(fp, 'w', encoding='utf-8'), indent=1)
    print('  d=%2d K=%d sp=%3d l1=%.3f %-9s seed=%d  base %.3f gate %.3f (dF1 %+.3f)'
          % (d, nc, sp, l1, tn, seed, fb, fg, fg - fb), flush=True)
    return rec


def main():
    print('[A-M2b] spread-calibrated clustered benchmark (published configs)')
    units = []
    for (d, nc, sp, l1) in CONFIGS:
        for (tn, rel) in RELS:
            for s in SEEDS:
                units.append(unit(d, nc, sp, l1, s, tn, rel))

    summary = {}
    for (d, nc, sp, l1) in CONFIGS:
        for (tn, rel) in RELS:
            u = [x for x in units if x['d'] == d and x['nc'] == nc
                 and x['sp'] == sp and x['rel'] == tn]
            df = np.array([x['d_f1'] for x in u])
            summary['d%d_k%d_%s' % (d, nc, tn)] = dict(
                rel=rel, n=len(u),
                d_f1_mean=round(float(df.mean()), 4),
                d_f1_sd=round(float(df.std(ddof=1)), 4) if len(u) > 1 else 0.0,
                n_pos=int((df > 0).sum()),
                f1_base=round(float(np.mean([x['f1_base'] for x in u])), 4),
                f1_gate=round(float(np.mean([x['f1_gate'] for x in u])), 4))
    json.dump({'summary': summary, 'units': units},
              open(os.path.join(OUT, 'spread_calib.json'), 'w', encoding='utf-8'), indent=1)

    print('\n### summary (dF1 = gate - base, mean over 10 seeds)')
    for k, v in summary.items():
        print('  %-16s rel=%.3f  dF1 %+.4f (sd %.4f)  %2d/%d pos  F1 base %.3f gate %.3f'
              % (k, v['rel'], v['d_f1_mean'], v['d_f1_sd'], v['n_pos'], v['n'],
                 v['f1_base'], v['f1_gate']))
    print('\nsaved -> %s' % os.path.join(OUT, 'spread_calib.json'))


if __name__ == '__main__':
    main()
