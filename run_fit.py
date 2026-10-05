# -*- coding: utf-8 -*-
"""End-to-end reproduction of the per-cancer result files.

run_all.py regenerates the FIGURES from the pre-computed data/ JSON.  run_fit.py
regenerates the data/ JSON itself: for each TCGA cancer type it fits the three arms
-- converged NOTEARS, CAGate base (gamma == 1) and CAGate gate -- from the UCSC Xena
expression matrix and writes data/pan_cancer/<CANCER>.json, so the numbers behind
Tables S5-S6 can be recomputed from the raw data and not only trusted.

The expression matrices are not redistributed.  Fetch them from UCSC Xena
(see download_tcga.py / https://xenabrowser.net) and point --data-dir at a directory
holding one TCGA_<CANCER>_HiSeqV2.tsv per cancer type, or pass --download to fetch
them with download_tcga.py.

Usage:
    python run_fit.py --data-dir ./tcga --d 100 CHOL DLBC     # named cancers
    python run_fit.py --data-dir ./tcga --d 100               # every present cancer
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from cagate import solve_cagate_lbfgsb
from notears_linear import notears_linear

L1, GA, K, THR = 0.1, 0.5, 5, 0.3
OUT = os.path.join(HERE, 'data', 'pan_cancer')
CANCERS = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM',
           'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC',
           'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD',
           'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']


def load(cancer, d, data_dir):
    """Read TCGA_<CANCER>_HiSeqV2.tsv, keep the top-d genes by variance, z-score."""
    p = os.path.join(data_dir, 'TCGA_%s_HiSeqV2.tsv' % cancer)
    if not os.path.exists(p):
        return None, None
    df = pd.read_csv(p, sep='\t', index_col=0)
    genes_all = list(df.index)
    data = df.values.astype(np.float64)
    var = data.var(axis=1)
    top = np.argsort(var)[-d:]
    genes = [genes_all[i] for i in top]
    X = data[top, :].T
    X = (X - X.mean(axis=0)) / np.clip(X.std(axis=0), 1e-6, None)
    return X, genes


def fit_one(cancer, d, data_dir, out_dir=OUT):
    X, genes = load(cancer, d, data_dir)
    if X is None:
        print('  %-5s: no local matrix, skipped' % cancer)
        return None
    n = X.shape[0]
    cid = KMeans(n_clusters=min(K, n), random_state=0, n_init=10).fit_predict(X)

    t0 = time.time()
    rb = solve_cagate_lbfgsb(X, cid=None, use_gate=False, l1=L1)
    tb = time.time() - t0
    t0 = time.time()
    rg = solve_cagate_lbfgsb(X, cid=cid, use_gate=True, l1=L1, gate_alpha=GA)
    tg = time.time() - t0
    t0 = time.time()
    Wn = notears_linear(X, lambda1=L1, loss_type='l2', max_iter=100, h_tol=1e-8, rho_max=1e16)
    tn = time.time() - t0

    rec = {'cancer': cancer, 'status': 'OK', 'n': int(n), 'd': int(X.shape[1]),
           'genes': genes,
           'edges': {'base': int(rb['n_edges']), 'gate': int(rg['n_edges']),
                     'notears': int((np.abs(Wn) > THR).sum())},
           'h': {'base': float(rb['h']), 'gate': float(rg['h'])},
           'secs': {'base': round(tb, 1), 'gate': round(tg, 1), 'notears': round(tn, 1)}}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, '%s.json' % cancer), 'w', encoding='utf-8') as f:
        json.dump(rec, f, indent=2)
    print('  %-5s: base=%d gate=%d notears=%d  (%.0fs/%.0fs/%.0fs)'
          % (cancer, rec['edges']['base'], rec['edges']['gate'],
             rec['edges']['notears'], tb, tg, tn), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', required=True,
                    help='directory holding TCGA_<CANCER>_HiSeqV2.tsv files')
    ap.add_argument('--d', type=int, default=100)
    ap.add_argument('--out', default=OUT, help='output directory (default: data/pan_cancer)')
    ap.add_argument('cancers', nargs='*', help='cancer abbreviations (default: all present)')
    a = ap.parse_args()
    todo = [c.upper() for c in a.cancers] or CANCERS
    print('fitting %d cancer type(s) at d=%d from %s' % (len(todo), a.d, a.data_dir))
    done = 0
    for c in todo:
        if fit_one(c, a.d, a.data_dir, a.out) is not None:
            done += 1
    print('done: %d/%d fitted' % (done, len(todo)))


if __name__ == '__main__':
    main()
