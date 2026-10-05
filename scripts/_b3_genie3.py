# -*- coding: utf-8 -*-
"""Published external baseline: GENIE3 (Huynh-Thu et al. 2010).

Uses the AUTHOR's official python implementation (vendored verbatim in
genie3_vendor/GENIE3.py, fetched from github.com/vahuynh/GENIE3).

Two comparisons, both checkpointed per unit:

  synth : on the clustered synthetic benchmark WITH GROUND TRUTH, rank edges by
          each method and compare AUPR + precision at the gate's own edge count.
          Methods: GENIE3 (RF, K=sqrt, ntrees=1000), CAGate gate, CAGate base,
          NOTEARS.
  real  : on every TCGA cancer in mega33_w/, compare the STRING>=700 support of
          each method's top-k edges (k = CAGate gate count).

Run:  python _b3_genie3.py --part synth|real|both
"""
import os, sys, json, gzip, time, argparse, pickle
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

# Paths are resolved relative to this file so the script runs from a fresh
# clone. External inputs (TCGA panels, STRING v12 raw files) are located via
# the environment variables documented in the repository README.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
C = SCRIPTS_DIR
REPO = REPO_ROOT
VDIR = os.environ.get('STRING_DATA_DIR', '')
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(SCRIPTS_DIR, 'genie3_vendor'))

import numpy as np
from _cfix_solver import gen_cluster, solve_cagate_lbfgsb, f1
from notears_linear import notears_linear
from GENIE3 import GENIE3 as genie3_run

OUT = os.path.join(DATA_DIR, 'genie3')
SYNTH = os.path.join(OUT, 'synth')
REAL = os.path.join(OUT, 'real')
WDIR = os.path.join(DATA_DIR, 'mega33_w')
THR = 0.3

CONFIGS = [(15, 5, 100, 0.01), (15, 8, 60, 0.01), (20, 6, 80, 0.005)]
NSEED = 10
KS_EVAL = [5, 8, 10, 15, 20, 25, 30, 40]

# cancers: published-Nature cohort names present in mega33_w
CANCERS = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM',
           'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC',
           'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD',
           'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']


# ── metrics ──────────────────────────────────────────────────────────────────
def aupr(scores, truth):
    """average precision of the ranked directed edge list (diagonal excluded)."""
    d = scores.shape[0]
    s = np.abs(scores.copy()); np.fill_diagonal(s, -np.inf)
    t = (np.abs(truth) > 0).astype(int); np.fill_diagonal(t, 0)
    s = s.ravel(); t = t.ravel()
    if t.sum() == 0:
        return 0.0
    order = np.argsort(-s)
    tt = t[order]
    tp = np.cumsum(tt)
    fp = np.cumsum(1 - tt)
    prec = tp / np.maximum(tp + fp, 1)
    return float(np.sum(prec * tt) / t.sum())


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


# ── synthetic ────────────────────────────────────────────────────────────────
def synth_unit(d, nc, sp, l1, seed, ntrees):
    fp = os.path.join(SYNTH, 'd%d_n%d_s%d_seed%d.json' % (d, nc, sp, seed))
    if os.path.exists(fp):
        return 'skip'
    X, Wt, cid = gen_cluster(d=d, nc=nc, sp=sp, seed=seed, transpose=False)
    np.random.seed(seed)          # sklearn RF with random_state=None uses the global RNG
    t0 = time.time()
    G = genie3_run(X, tree_method='RF', K='sqrt', ntrees=ntrees, nthreads=1)
    tg = time.time() - t0
    rb = solve_cagate_lbfgsb(X, cid=None, use_gate=False, l1=l1)
    rg = solve_cagate_lbfgsb(X, cid=cid, use_gate=True, l1=l1, gate_alpha=0.5)
    Wn = notears_linear(X, lambda1=l1, loss_type='l2', max_iter=100,
                        h_tol=1e-8, rho_max=1e16)
    arms = {'genie3': G, 'gate': rg['W'], 'base': rb['W'], 'notears': Wn}
    kg = n_edges(rg['W'])
    rec = {'d': d, 'nc': nc, 'sp': sp, 'l1': l1, 'seed': seed, 'ntrees': ntrees,
           'k_gate': kg, 'genie3_secs': round(tg, 1), 'arms': {}}
    for a, W in arms.items():
        rec['arms'][a] = {
            'aupr': round(aupr(W, Wt), 5),
            'edges@0.3': n_edges(W),
            'f1@0.3': round(f1(W, Wt)[0], 5),
            'prec@k_gate': round(prec_at_k(W, Wt, min(kg, d * d - d)), 5),
            'prec_curve': {str(k): round(prec_at_k(W, Wt, min(k, d * d - d)), 5)
                           for k in KS_EVAL}}
    os.makedirs(SYNTH, exist_ok=True)
    json.dump(rec, open(fp, 'w', encoding='utf-8'), indent=2)
    print('  synth d=%d nc=%d sp=%d seed=%d: GENIE3 AUPR=%.3f gate=%.3f base=%.3f NOTEARs=%.3f'
          % (d, nc, sp, seed, rec['arms']['genie3']['aupr'], rec['arms']['gate']['aupr'],
             rec['arms']['base']['aupr'], rec['arms']['notears']['aupr']), flush=True)
    return 'ok'


# ── real: STRING pool (cached) ───────────────────────────────────────────────
def string_pool(our_genes):
    cache = os.path.join(C, 'b3_cache', 'string_pairs.pkl')
    if os.path.exists(cache):
        with open(cache, 'rb') as fh:
            return pickle.load(fh)
    from collections import defaultdict
    s2e = defaultdict(set)
    with gzip.open(os.path.join(VDIR, 'string_info.txt.gz'), 'rt',
                   encoding='utf-8', errors='ignore') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) >= 2:
                s2e[p[1].upper()].add(p[0].split('.', 1)[-1])
    e2s = {}
    for s, es in s2e.items():
        for e in es:
            e2s.setdefault(e, s)
    pool = set()
    with gzip.open(os.path.join(VDIR, 'string_ppi_full.txt.gz'), 'rt',
                   encoding='utf-8', errors='ignore') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\n').split()
            if len(p) < 3:
                continue
            try:
                sc = int(p[-1])
            except ValueError:
                continue
            if sc < 700:
                continue
            sa = e2s.get(p[0].split('.', 1)[-1]); sb = e2s.get(p[1].split('.', 1)[-1])
            if sa and sb and sa != sb:
                pool.add((sa, sb)); pool.add((sb, sa))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, 'wb') as fh:
        pickle.dump(pool, fh)
    print('  STRING pool cached: %d directed pairs' % len(pool), flush=True)
    return pool


def string_support(genes, W, pool, k):
    G = [g.upper() for g in genes]
    d = len(G)
    s = np.abs(W.copy()); np.fill_diagonal(s, 0)
    flat = np.argsort(-s.ravel())[:min(k, d * d - d)]
    tot = 0
    for f in flat:
        i, j = divmod(int(f), d)
        if (G[i], G[j]) in pool:
            tot += 1
    return round(100.0 * tot / max(1, len(flat)), 2), len(flat)


def real_unit(cancer, pool, ntrees, nthreads):
    fp = os.path.join(REAL, '%s.json' % cancer)
    if os.path.exists(fp):
        return 'skip'
    npz = os.path.join(WDIR, '%s.npz' % cancer)
    js = os.path.join(WDIR, '%s.json' % cancer)
    if not (os.path.exists(npz) and os.path.exists(js)):
        print('  %-5s: mega33_w missing, skip' % cancer, flush=True)
        return 'miss'
    meta = json.load(open(js, encoding='utf-8'))
    if meta.get('status') != 'OK':
        return 'miss'
    genes = meta['genes']
    z = np.load(npz, allow_pickle=True)
    Wb, Wg, Wn = z['base'], z['gate'], z['notears']
    # GENIE3 needs the expression matrix: cache it from run_fit loader
    from _b3_runfit import load as rf_load
    X, g2 = rf_load(cancer, len(genes))
    if X is None:
        print('  %-5s: no raw matrix for GENIE3, skip' % cancer, flush=True)
        return 'miss'
    np.random.seed(0)
    t0 = time.time()
    G = genie3_run(X, tree_method='RF', K='sqrt', ntrees=ntrees, nthreads=nthreads)
    tg = time.time() - t0
    kg = n_edges(Wg)
    rec = {'cancer': cancer, 'n': int(X.shape[0]), 'd': int(X.shape[1]),
           'k_gate': kg, 'ntrees': ntrees, 'genie3_secs': round(tg, 1), 'arms': {}}
    for a, W in (('genie3', G), ('gate', Wg), ('base', Wb), ('notears', Wn)):
        pct, ne = string_support(genes, W, pool, kg)
        rec['arms'][a] = {'string_pct@k_gate': pct, 'edges_evaluated': ne,
                          'edges@0.3': n_edges(W)}
    os.makedirs(REAL, exist_ok=True)
    json.dump(rec, open(fp, 'w', encoding='utf-8'), indent=2)
    print('  %-5s: n=%d k=%d  GENIE3 %.1f%%  gate %.1f%%  base %.1f%%  NOTEARS %.1f%%  (%.0fs)'
          % (cancer, X.shape[0], kg, rec['arms']['genie3']['string_pct@k_gate'],
             rec['arms']['gate']['string_pct@k_gate'], rec['arms']['base']['string_pct@k_gate'],
             rec['arms']['notears']['string_pct@k_gate'], tg), flush=True)
    return 'ok'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', default='both', choices=['synth', 'real', 'both'])
    ap.add_argument('--ntrees-synth', type=int, default=1000)
    ap.add_argument('--ntrees-real', type=int, default=1000)
    ap.add_argument('--nthreads-real', type=int, default=8)
    a = ap.parse_args()
    print('JOB B / GENIE3 external baseline  part=%s  ntrees_synth=%d ntrees_real=%d '
          'nthreads_real=%d' % (a.part, a.ntrees_synth, a.ntrees_real, a.nthreads_real),
          flush=True)
    if a.part in ('synth', 'both'):
        os.makedirs(SYNTH, exist_ok=True)
        print('-- synthetic (ground truth) --', flush=True)
        for (d, nc, sp, l1) in CONFIGS:
            for seed in range(NSEED):
                try:
                    synth_unit(d, nc, sp, l1, seed, a.ntrees_synth)
                except Exception as e:
                    print('  synth d=%d nc=%d s=%d ERROR %s %s'
                          % (d, nc, seed, type(e).__name__, e), flush=True)
    if a.part in ('real', 'both'):
        os.makedirs(REAL, exist_ok=True)
        print('-- real TCGA (STRING support) --', flush=True)
        pool = string_pool(set())
        for c in CANCERS:
            try:
                real_unit(c, pool, a.ntrees_real, a.nthreads_real)
            except Exception as e:
                print('  %-5s ERROR %s %s' % (c, type(e).__name__, e), flush=True)
    print('JOB B done', flush=True)


if __name__ == '__main__':
    main()
