"""Mechanistic diagnosis: why is the per-subtype mean gate pinned at 0.50?

gate_g = 1/(1+exp(-alpha*(sigma_bar - sigma_g))).
If sigma_k (per-cluster residual-energy dispersion) is nearly identical across
clusters, the gate is a constant no matter which labels you feed it.

We recompute sigma_k for the published subtypes and report the implied gate at
several alpha, to separate "wrong alpha" from "no signal to gate on".
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
DATA_DIR = os.path.join(_REPO, 'data')
STRING_DATA_DIR = os.environ.get('STRING_DATA_DIR', '')
GO_DATA_DIR = os.environ.get('GO_DATA_DIR', STRING_DATA_DIR)
TCGA_DATA_DIR = os.environ.get('TCGA_DATA_DIR', '')
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)
import os, sys, json, re
pass  # paths resolved via the header block
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from _cfix_solver import cluster_gate_np

DATA = TCGA_DATA_DIR
SUB = os.path.join(DATA_DIR, 'subtypes')
RUN = os.path.join(DATA_DIR, 'subtypes_run')
ALIAS = {'COAD': 'COADREAD', 'READ': 'COADREAD'}


def norm(s):
    s = str(s).strip()
    m = re.match(r'(TCGA-[0-9A-Za-z]{2}-[0-9A-Za-z]{4})', s)
    return m.group(1) if m else s[:15]


def load_with_ids(cancer, n_genes=100):
    df = pd.read_csv(os.path.join(DATA, 'TCGA_%s_HiSeqV2.tsv' % cancer), sep='\t', index_col=0)
    ga = list(df.index); data = df.values.astype(np.float64)
    top = np.argsort(data.var(axis=1))[-n_genes:]
    X = data[top, :].T
    X = (X - X.mean(0)) / np.clip(X.std(0), 1e-6, None)
    return X, [ga[i] for i in top], list(df.columns)


def main():
    for cancer, col in [('BRCA', 'PAM50Call_RNAseq'), ('GBM', 'GeneExp_Subtype'),
                        ('LUAD', 'Expression_Subtype')]:
        X, genes, ids = load_with_ids(cancer)
        clin = pd.read_csv(os.path.join(SUB, '%s_clinical.tsv' % ALIAS.get(cancer, cancer)),
                           sep='\t', low_memory=False)
        lab = clin[[clin.columns[0], col]].copy(); lab.columns = ['sid', 'lab']
        lab['k'] = lab['sid'].map(norm)
        lab = lab.dropna(subset=['lab'])
        lab = lab[lab['lab'].astype(str).str.lower() != 'nan']
        m = dict(zip(lab['k'], lab['lab'].astype(str)))
        keep = [(i, m[norm(s)]) for i, s in enumerate(ids) if norm(s) in m]
        idx = [i for i, _ in keep]; y = [v for _, v in keep]
        cats = sorted(set(y)); cid = np.array([cats.index(v) for v in y], dtype=int)
        Xs = X[idx, :]

        z = np.load(os.path.join(RUN, '%s.npz' % cancer))
        # diagnostic: use the published-label gate W
        for arm in ['base', 'gate']:
            W = z[arm]
            R = Xs - Xs @ W
            res = (R ** 2).mean(axis=1)
            nc = len(cats)
            sigma = np.array([res[cid == g].std(ddof=1) if (cid == g).sum() > 1 else 0.0
                              for g in range(nc)])
            pos = sigma[sigma > 0]
            sbar = float(np.median(pos)) if pos.size else 1.0
            print('%-5s [%-6s] sigma per subtype:' % (cancer, arm),
                  dict(zip(cats, np.round(sigma, 5))))
            print('        spread=%.5f (max-min)  sigma_bar=%.5f  rel=%.2e'
                  % (sigma.max() - sigma.min(), sbar,
                     (sigma.max() - sigma.min()) / sbar))
            for a in [0.5, 5, 50, 500]:
                gpc = 1.0 / (1.0 + np.exp(-a * (sbar - sigma)))
                print('        alpha=%-5g gate/type: %s' % (a, np.round(gpc, 4).tolist()))
            # what alpha would create a 0.2 gate spread?
            need = sigma.max() - sigma.min()
            print('        -> to separate gate by 0.2 requires alpha ~ %.0f' % (4.0 / need if need > 0 else float('inf')))
            print()


if __name__ == '__main__':
    main()
