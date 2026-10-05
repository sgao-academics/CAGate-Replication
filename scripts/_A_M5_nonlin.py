# -*- coding: utf-8 -*-
"""A-M5: synthetic benchmark with hub-dominated topology, with and without
nonlinear mechanisms.

The clustered benchmark of the main text is a linear Gaussian SEM over an
Erdos-Renyi graph.  Two objections follow: a random-forest method is handicapped
by construction on linear Gaussian data, and a degree-uniform graph does not look
like a regulatory network, which is hub-dominated.  This script repeats the
four-arm comparison on a benchmark that removes both features:

  topology   : preferential attachment in a topological order, so the *out*-degree
               distribution is heavy-tailed -- a few regulators carry most edges,
               as transcription factors do;
  mechanisms : every edge passes through one of four link functions (identity,
               tanh, sine, and a signed square), each contribution rescaled to
               unit variance so that no coordinate blows up; the linear control
               (all links identity) is run on exactly the same topology and seeds,
               so nonlinearity and topology can be separated;
  clusters   : K clusters differing in noise dispersion, as in the main-text
               benchmark, so the gate still has a dispersion axis to read.

Metrics are defined identically for all arms: AUPR of the ranked edge list
against the true DAG, and precision at the gate's own edge count.  Convergence
(the acyclicity residual h) is recorded for every arm so a low score can be
traced to the optimiser rather than to the method.

Run:  python scripts/_A_M5_nonlin.py
Writes data/nonlin/nonlin.json plus one checkpoint per unit.
"""
import os, sys, json, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(SCRIPTS_DIR, 'genie3_vendor'))

import numpy as np
from _cfix_solver import solve_cagate_lbfgsb, f1, h_dag
from notears_linear import notears_linear
from GENIE3 import GENIE3 as genie3_run

OUT = os.path.join(DATA_DIR, 'nonlin')
os.makedirs(OUT, exist_ok=True)
THR = 0.3
LINKS = ('identity', 'tanh', 'sine', 'square')


def log(*a):
    print(*a, flush=True)


# ── generator ────────────────────────────────────────────────────────────────
def gen_sf(d=40, nc=5, sp=100, m=3, seed=0, nonlinear=True):
    """Scale-free DAG over d variables in nc noise-dispersion clusters.

    Returns X, Wt (signed true weights), cid, out-degree vector, and the link
    function chosen per edge.  Each edge contribution is rescaled to unit
    variance, so no coordinate explodes and the signal-to-noise ratio is
    comparable across seeds.
    """
    rng = np.random.RandomState(seed)
    Wt = np.zeros((d, d))
    links = {}
    outdeg = np.zeros(d)
    for j in range(1, d):
        cand = np.arange(j)
        p = outdeg[cand] + 1.0
        p = p / p.sum()
        k = min(m, j)
        parents = rng.choice(cand, size=k, replace=False, p=p)
        for i in parents:
            w = rng.uniform(1.0, 2.0) * rng.choice([-1, 1])
            Wt[i, j] = w
            links[(i, j)] = LINKS[rng.randint(len(LINKS))] if nonlinear else 'identity'
            outdeg[i] += 1

    n = nc * sp
    X = np.zeros((n, d))
    cid = np.zeros(n, dtype=int)
    for c in range(nc):
        scale = 0.3 + 2.0 * c / (nc - 1) if nc > 1 else 0.3
        idx = slice(c * sp, (c + 1) * sp)
        Z = scale * rng.randn(sp, d)
        Xc = np.zeros((sp, d))
        for j in range(d):
            acc = np.zeros(sp)
            for i in range(j):
                if Wt[i, j] != 0:
                    v = Wt[i, j] * Xc[:, i]
                    g = links[(i, j)]
                    if g == 'tanh':
                        u = np.tanh(v)
                    elif g == 'sine':
                        u = np.sin(v)
                    elif g == 'square':
                        u = np.sign(v) * v ** 2
                    else:
                        u = v
                    s = u.std()
                    acc += u / s if s > 1e-9 else u
            Xc[:, j] = acc + Z[:, j]
        X[idx] = Xc
        cid[idx] = c
    return X, Wt, cid, outdeg, links


def aupr(scores, truth):
    d = scores.shape[0]
    s = np.abs(scores.copy()); np.fill_diagonal(s, -np.inf)
    t = (np.abs(truth) > 0).astype(int); np.fill_diagonal(t, 0)
    s = s.ravel(); t = t.ravel()
    if t.sum() == 0:
        return 0.0
    tt = t[np.argsort(-s)]
    tp = np.cumsum(tt); fp = np.cumsum(1 - tt)
    return float(np.sum((tp / np.maximum(tp + fp, 1)) * tt) / t.sum())


def prec_at_k(scores, truth, k):
    d = scores.shape[0]
    s = np.abs(scores.copy()); np.fill_diagonal(s, 0)
    t = (np.abs(truth) > 0).astype(int); np.fill_diagonal(t, 0)
    flat = np.argsort(-s.ravel())[:k]
    sel = np.zeros(d * d, dtype=int); sel[flat] = 1; sel = sel.reshape(d, d)
    return float((sel & t).sum()) / max(1, k)


def n_edges(W, thr=THR):
    W = np.abs(W.copy()); np.fill_diagonal(W, 0)
    return int((W > thr).sum())


def unit(d, nc, sp, m, seed, nonlinear, ntrees=500, l1=0.05):
    tag = 'nonlin' if nonlinear else 'linear'
    fp = os.path.join(OUT, '%s_d%d_n%d_s%d_m%d_seed%d.json' % (tag, d, nc, sp, m, seed))
    if os.path.exists(fp):
        return json.load(open(fp, encoding='utf-8'))
    X, Wt, cid, outdeg, links = gen_sf(d=d, nc=nc, sp=sp, m=m, seed=seed, nonlinear=nonlinear)
    ne = int((np.abs(Wt) > 0).sum())
    order = np.sort(outdeg)[::-1]
    np.random.seed(seed)
    t0 = time.time()
    G = genie3_run(X, tree_method='RF', K='sqrt', ntrees=ntrees, nthreads=1)
    tg = time.time() - t0
    rb = solve_cagate_lbfgsb(X, cid=None, use_gate=False, l1=l1)
    rg = solve_cagate_lbfgsb(X, cid=cid, use_gate=True, l1=l1, gate_alpha=0.5)
    Wn = notears_linear(X, lambda1=l1, loss_type='l2', max_iter=100, h_tol=1e-8, rho_max=1e16)
    arms = {'genie3': G, 'gate': rg['W'], 'base': rb['W'], 'notears': Wn}
    hs = {'gate': float(rg['h']), 'base': float(rb['h']),
          'notears': float(h_dag(Wn)[0] if np.isfinite(Wn).all() else np.inf)}
    kg = n_edges(rg['W'])
    rec = {'d': d, 'nc': nc, 'sp': sp, 'm': m, 'seed': seed, 'nonlinear': bool(nonlinear),
           'n': int(X.shape[0]), 'true_edges': ne,
           'max_outdeg': int(outdeg.max()), 'mean_outdeg': float(outdeg.mean()),
           'n_regulators_carrying_half': int(order.cumsum().searchsorted(ne / 2) + 1),
           'col_std_min': float(X.std(0).min()), 'col_std_max': float(X.std(0).max()),
           'gate_edges': kg, 'h': hs, 'genie3_secs': round(tg, 1)}
    for a, W in arms.items():
        f, p, r = f1(W, Wt, THR)
        rec[a] = dict(aupr=round(aupr(W, Wt), 4), f1=round(f, 4),
                      prec_at_k=round(prec_at_k(W, Wt, kg), 4), edges=n_edges(W))
    json.dump(rec, open(fp, 'w', encoding='utf-8'), indent=1)
    log('  %-6s d=%d seed=%d  true=%d maxoutdeg=%d | AUPR gate %.3f genie3 %.3f base %.3f notears %.3f | h gate %.1e'
        % (tag, d, seed, ne, outdeg.max(), rec['gate']['aupr'], rec['genie3']['aupr'],
           rec['base']['aupr'], rec['notears']['aupr'], hs['gate']))
    return rec


def main():
    log('[A-M5] hub-dominated synthetic benchmark (linear control + nonlinear)')
    units = []
    for nonlinear in (False, True):
        for seed in range(3):
            units.append(unit(40, 5, 100, 3, seed, nonlinear))
    log('\n### summary')
    s = {}
    for tag, nl in (('linear', False), ('nonlinear', True)):
        u = [x for x in units if x['nonlinear'] == nl]
        log('  %s: true edges %.0f | max out-degree %.0f | regulators carrying half the edges: %.1f'
            % (tag, np.mean([x['true_edges'] for x in u]),
               np.mean([x['max_outdeg'] for x in u]),
               np.mean([x['n_regulators_carrying_half'] for x in u])))
        row = {}
        for a in ('genie3', 'gate', 'base', 'notears'):
            au = float(np.mean([x[a]['aupr'] for x in u]))
            pk = float(np.mean([x[a]['prec_at_k'] for x in u]))
            fv = float(np.mean([x[a]['f1'] for x in u]))
            ee = float(np.mean([x[a]['edges'] for x in u]))
            row[a] = dict(aupr=round(au, 4), prec_at_k=round(pk, 4), f1=round(fv, 4),
                          edges=round(ee, 1))
            log('    %-8s AUPR %.3f  prec@k %.3f  F1 %.3f  edges %.0f' % (a, au, pk, fv, ee))
        row['gate_vs_base_aupr'] = round(float(np.mean(
            [x['gate']['aupr'] - x['base']['aupr'] for x in u])), 4)
        row['gate_vs_genie3_aupr'] = round(float(np.mean(
            [x['gate']['aupr'] - x['genie3']['aupr'] for x in u])), 4)
        row['n_units'] = len(u)
        s[tag] = row
    json.dump({'summary': s, 'units': units},
              open(os.path.join(OUT, 'nonlin.json'), 'w', encoding='utf-8'), indent=1)
    log('\nsaved -> %s' % os.path.join(OUT, 'nonlin.json'))


if __name__ == '__main__':
    main()
