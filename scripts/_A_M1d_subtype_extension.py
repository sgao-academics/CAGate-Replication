# -*- coding: utf-8 -*-
"""A-M1d: the published-subtype back-annotation, extended to GBM and LUAD.

Note SN25 back-annotates the K-means partition against the published subtype
calls on TCGA-BRCA only (ARI = 0.35, and a per-group residual dispersion that is
wider under K-means than under PAM50).  Reviewer B-M2 asks whether the same two
facts hold where other published calls exist, so the identical procedure is run
for GBM (Verhaak GeneExp_Subtype) and LUAD (TCGA Expression_Subtype).

This reuses the exact analysis of ``_A_M1_subtype.py``, with the label column
and the cohort taken as parameters.  Reads TCGA matrices, the shipped subtype
tables, and the base arm of ``subtypes_run/<cancer>.npz``.  Writes
A_M1_subtype_ext.json.  Reproduces the BRCA numbers as a self-check.
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
DATA_DIR = os.path.join(_REPO, 'data')
TCGA_DATA_DIR = os.environ.get('TCGA_DATA_DIR', '')

sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)
import json, re

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

DATA = TCGA_DATA_DIR
SUB = os.path.join(DATA_DIR, 'subtypes')
RUN = os.path.join(DATA_DIR, 'subtypes_run')
OUT = os.path.join(DATA_DIR, 'A_M1_subtype_ext.json')

# cohort -> (published-label column in the cBioPortal table, table stem)
CAN = [
    ('BRCA', 'PAM50Call_RNAseq', 'BRCA'),      # Parker et al. 2009
    ('GBM',  'GeneExp_Subtype',  'GBM'),       # Verhaak et al. 2010
    ('LUAD', 'Expression_Subtype', 'LUAD'),    # Wilkerson et al. 2012
]


def log(*a):
    print(*a, flush=True)


def norm(s):
    m = re.match(r'(TCGA-[0-9A-Za-z]{2}-[0-9A-Za-z]{4})', str(s).strip())
    return m.group(1) if m else str(s)[:15]


def run_one(cancer, col, stem):
    df = pd.read_csv(os.path.join(DATA, 'TCGA_%s_HiSeqV2.tsv' % cancer),
                     sep='\t', index_col=0)
    genes_all = list(df.index)
    data = df.values.astype(np.float64)
    top = np.argsort(data.var(axis=1))[-100:]
    genes = [genes_all[i] for i in top]
    X = data[top, :].T
    X = (X - X.mean(0)) / np.clip(X.std(0), 1e-6, None)
    ids = list(df.columns)

    clin = pd.read_csv(os.path.join(SUB, '%s_clinical.tsv' % stem),
                       sep='\t', low_memory=False)
    lab = clin[[clin.columns[0], col]].copy()
    lab.columns = ['sid', 'lab']
    lab['k'] = lab['sid'].map(norm)
    lab = lab.dropna(subset=['lab'])
    lab = lab[lab['lab'].astype(str).str.lower() != 'nan']
    m = dict(zip(lab['k'], lab['lab'].astype(str)))
    keep = [(i, m[norm(s)]) for i, s in enumerate(ids) if norm(s) in m]
    idx = [i for i, _ in keep]
    y = [v for _, v in keep]
    cats = sorted(set(y))
    cid_p = np.array([cats.index(v) for v in y], dtype=int)
    Xs = X[idx, :]
    log('[%s] %d/%d samples with %s -> %d subtypes %s'
        % (cancer, len(idx), len(ids), col, len(cats), cats))

    cid_km = KMeans(n_clusters=len(cats), random_state=0, n_init=10).fit_predict(Xs)
    ari = adjusted_rand_score(cid_p, cid_km)

    W = np.load(os.path.join(RUN, '%s.npz' % cancer))['base']
    R = Xs - Xs @ W
    res = (R ** 2).mean(axis=1)

    def sigmas(cid):
        out = []
        for g in range(int(cid.max()) + 1):
            v = res[cid == g]
            out.append(float(v.std(ddof=1)) if v.size > 1 else 0.0)
        return np.array(out)

    s_p, s_km = sigmas(cid_p), sigmas(cid_km)
    rel_p = (s_p.max() - s_p.min()) / np.median(s_p)
    rel_k = (s_km.max() - s_km.min()) / np.median(s_km)
    log('  ARI=%.3f | PAM50/pub sigma %s spread=%.4f rel=%.3f | K-means sigma %s spread=%.4f rel=%.3f'
        % (ari, np.round(s_p, 4).tolist(), s_p.max() - s_p.min(), rel_p,
           np.round(s_km, 4).tolist(), s_km.max() - s_km.min(), rel_k))
    return dict(cancer=cancer, subtype_col=col, cats=cats, n=int(len(idx)),
                n_total=int(len(ids)), ari=float(ari),
                pub_sizes=np.bincount(cid_p).tolist(),
                km_sizes=np.bincount(cid_km).tolist(),
                pub_sigma=s_p.tolist(), km_sigma=s_km.tolist(),
                pub_spread=float(s_p.max() - s_p.min()), km_spread=float(s_km.max() - s_km.min()),
                pub_rel_spread=float(rel_p), km_rel_spread=float(rel_k))


def main():
    out = {}
    for cancer, col, stem in CAN:
        out[cancer] = run_one(cancer, col, stem)
    json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=1)
    log('\nsaved -> %s' % OUT)


if __name__ == '__main__':
    main()
