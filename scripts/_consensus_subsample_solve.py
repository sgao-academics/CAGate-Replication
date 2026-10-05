# -*- coding: utf-8 -*-
"""Cross-cancer consensus pipeline, step 1: subsample and solve.

For every TCGA cohort, z-score the genes of the fixed 300-gene KEGG hsa05200
panel, draw a random subsample of n = 500 samples and solve the UNGATED problem
(NOTEARS, gamma_k = 1) at |W| > 0.3, exactly as the cross-cancer analysis of the
manuscript (Table S7, Note SN28).  The subsample seed is a parameter so that the
recurrence ladder can be rebuilt from more than one draw.

Paths are resolved relative to this file so the script runs from a fresh clone.
External inputs are located via the environment variables documented in the
repository README.

Environment:
  TCGA_DATA_DIR    directory holding TCGA_<CANCER>_HiSeqV2.tsv (required)
  OUTDIR           directory for the per-cohort weight matrices (default
                   data/consensus_w; relative paths resolve against the repo)
  SUBSAMPLE_SEED   RNG seed of the n = 500 subsample (default 0)
  MAXN             subsample size (default 500)
  PANEL_D          number of panel genes to use (default 300)
  PANEL_FILE       optional JSON with a "panel" list, overriding the shipped panel

Usage:
  python scripts/_consensus_subsample_solve.py BRCA GBM LUAD
"""
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _cfix_solver import solve_cagate_lbfgsb  # noqa: E402

TCGA = os.environ.get('TCGA_DATA_DIR', os.path.join(REPO, 'data', 'tcga'))
OUT = os.environ.get('OUTDIR', os.path.join(REPO, 'data', 'consensus_w'))
if not os.path.isabs(OUT):
    OUT = os.path.join(REPO, OUT)
MAXN = int(os.environ.get('MAXN', '500'))
DD = int(os.environ.get('PANEL_D', '300'))
SEED = int(os.environ.get('SUBSAMPLE_SEED', '0'))

PF = os.environ.get('PANEL_FILE', '')
if PF and os.path.exists(PF):
    PANEL = [g.upper() for g in json.load(open(PF, encoding='utf-8'))['panel']][:DD]
else:
    # The shipped edge file carries the panel actually used by the manuscript.
    PANEL = [g.upper() for g in json.load(
        open(os.path.join(REPO, 'data', 'edges_d300.json'), encoding='utf-8'))['genes']][:DD]
PANEL_SET = set(PANEL)
D = len(PANEL)


def load_cancer(c):
    fn = os.path.join(TCGA, 'TCGA_%s_HiSeqV2.tsv' % c)
    if not os.path.exists(fn):
        return None
    rows = {}
    with open(fn, encoding='utf-8', errors='replace') as f:
        hdr = f.readline().rstrip('\n').split('\t')
        for ln in f:
            p = ln.rstrip('\n').split('\t')
            g = p[0].strip().upper()
            if g in PANEL_SET:
                rows[g] = p[1:]
    if len(rows) < D * 0.9:
        print('  [warn] only %d/%d panel genes present' % (len(rows), D))
    X = np.zeros((D, len(hdr) - 1), dtype=np.float64)
    for i, g in enumerate(PANEL):
        v = rows.get(g)
        if v is None:
            continue
        arr = np.array([np.nan if (s == '' or s.upper() == 'NA') else float(s) for s in v])
        arr = np.nan_to_num(arr, nan=np.nanmedian(arr) if np.isfinite(arr).any() else 0.0)
        X[i] = arr
    X = X.T
    n = X.shape[0]
    if n > MAXN:
        idx = np.random.RandomState(SEED).choice(n, MAXN, replace=False)
        X = X[idx]
    return X


def main(cancers):
    os.makedirs(OUT, exist_ok=True)
    print('panel d=%d  subsample n<=%d  seed=%d  out=%s' % (D, MAXN, SEED, OUT))
    for c in cancers:
        t0 = time.time()
        X = load_cancer(c)
        if X is None:
            print('[%s] tsv missing' % c)
            continue
        n = X.shape[0]
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
        hs = 1e-6 if n <= 100 else 0.1
        out = solve_cagate_lbfgsb(Xs, cid=None, use_gate=False, l1=hs,
                                  w_threshold=0.3, max_outer=40, max_inner=20,
                                  maxiter=1000)
        W = out['W']
        edges = int((np.abs(W) > 0.3).sum())
        np.savez(os.path.join(OUT, '%s.npz' % c), W=W.astype(np.float32),
                 genes=np.array(PANEL), n_used=n, seed=SEED)
        print('[%s] seed=%d d=%d n=%d edges=%d h=%.1e  %.1fs'
              % (c, SEED, D, n, edges, out['h'], time.time() - t0), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
