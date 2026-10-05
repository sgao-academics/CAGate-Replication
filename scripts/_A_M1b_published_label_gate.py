"""Does the cluster gate still help when the cluster labels are a PUBLISHED
molecular subtype (PAM50 / Verhaak / TCGA expression subtype) instead of a
K-means partition of the same expression matrix?

If the gain was an artefact of "clustering the data you then fit", swapping in
an EXTERNAL, independently-published label must destroy it.  If the gain
survives, the gate is capturing real, externally-defined biological
heterogeneity.

Arms (all on the identical sample subset that carries an official label):
  base        : no gate
  gate        : gate with the published subtype labels
  gate_km     : gate with K-means on the same data, same #clusters
  gate_perm   : gate with the published labels randomly permuted (sizes fixed)

Usage: _cfix_subtype_gate.py <CANCER> <SUBTYPE_COL> [n_genes] [l1] [gate_alpha]
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
import os, sys, json, re, time
pass  # paths resolved via the header block
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from _cfix_solver import solve_cagate_lbfgsb
from sklearn.cluster import KMeans

DATA = TCGA_DATA_DIR
SUB = os.path.join(DATA_DIR, 'subtypes')
OUT = os.path.join(DATA_DIR, 'subtypes_run')
os.makedirs(OUT, exist_ok=True)
ALIAS = {'COAD': 'COADREAD', 'READ': 'COADREAD'}


def norm(s):
    s = str(s).strip()
    m = re.match(r'(TCGA-[0-9A-Za-z]{2}-[0-9A-Za-z]{4})', s)
    return m.group(1) if m else s[:15]


def load_with_ids(cancer, n_genes=100):
    p = os.path.join(DATA, 'TCGA_%s_HiSeqV2.tsv' % cancer)
    df = pd.read_csv(p, sep='\t', index_col=0)
    genes_all = list(df.index)
    data = df.values.astype(np.float64)
    var = data.var(axis=1)
    top = np.argsort(var)[-n_genes:]
    genes = [genes_all[i] for i in top]
    X = data[top, :].T
    X = (X - X.mean(axis=0)) / np.clip(X.std(axis=0), 1e-6, None)
    return X, genes, list(df.columns)


def main():
    cancer = sys.argv[1].upper()
    col = sys.argv[2]
    n_genes = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    l1 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
    ga = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5

    sub_npz = os.path.join(OUT, '%s.npz' % cancer)
    sub_js = os.path.join(OUT, '%s.json' % cancer)
    if os.path.exists(sub_npz) and os.path.exists(sub_js):
        print('exists, skip:', sub_npz); return

    X, genes, ids = load_with_ids(cancer, n_genes)
    clin = pd.read_csv(os.path.join(SUB, '%s_clinical.tsv' % ALIAS.get(cancer, cancer)),
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
    Xs = X[idx, :]
    # integer labels, sorted for determinism
    cats = sorted(set(y))
    cid = np.array([cats.index(v) for v in y], dtype=int)
    nsub = len(cats)
    print('%s: %d/%d samples with %s -> %d clusters %s'
          % (cancer, len(idx), len(ids), col, nsub, cats), flush=True)
    print('   cluster sizes:', np.bincount(cid).tolist(), flush=True)

    rng = np.random.RandomState(0)
    cid_perm = cid[rng.permutation(len(cid))]
    cid_km = KMeans(n_clusters=nsub, random_state=0, n_init=10).fit_predict(Xs)

    arms = {}
    t0 = time.time()
    rb = solve_cagate_lbfgsb(Xs, cid=None, use_gate=False, l1=l1)
    arms['base'] = rb
    print('  base      edges=%-4d h=%.1e (%.0fs)' % (rb['n_edges'], rb['h'], time.time() - t0), flush=True)

    t0 = time.time()
    rg = solve_cagate_lbfgsb(Xs, cid=cid, use_gate=True, l1=l1, gate_alpha=ga)
    arms['gate'] = rg
    print('  gate[%s] edges=%-4d h=%.1e (%.0fs)' % (col[:14], rg['n_edges'], rg['h'], time.time() - t0), flush=True)

    t0 = time.time()
    rk = solve_cagate_lbfgsb(Xs, cid=cid_km, use_gate=True, l1=l1, gate_alpha=ga)
    arms['gate_km'] = rk
    print('  gate_km   edges=%-4d h=%.1e (%.0fs)' % (rk['n_edges'], rk['h'], time.time() - t0), flush=True)

    t0 = time.time()
    rp = solve_cagate_lbfgsb(Xs, cid=cid_perm, use_gate=True, l1=l1, gate_alpha=ga)
    arms['gate_perm'] = rp
    print('  gate_perm edges=%-4d h=%.1e (%.0fs)' % (rp['n_edges'], rp['h'], time.time() - t0), flush=True)

    np.savez_compressed(sub_npz,
                        base=arms['base']['W'].astype(np.float32),
                        gate=arms['gate']['W'].astype(np.float32),
                        gate_km=arms['gate_km']['W'].astype(np.float32),
                        gate_perm=arms['gate_perm']['W'].astype(np.float32))

    # per-cluster mean gate (biological interpretability)
    cg = {}
    for g in range(nsub):
        msk = cid == g
        cg[cats[g]] = round(float(rg['gates'][msk].mean()), 4)

    json.dump({'cancer': cancer, 'subtype_col': col, 'n_total': int(len(ids)),
               'n_subset': int(len(idx)), 'n_clusters': nsub, 'clusters': cats,
               'cluster_sizes': np.bincount(cid).tolist(),
               'genes': genes, 'l1': l1, 'gate_alpha': ga, 'n_genes': n_genes,
               'edges': {a: int(arms[a]['n_edges']) for a in arms},
               'h': {a: float(arms[a]['h']) for a in arms},
               'gate_mean_per_cluster': cg},
              open(sub_js, 'w', encoding='utf-8'), indent=2)
    print('  ->', sub_npz, flush=True)
    print('  edges:', {a: arms[a]['n_edges'] for a in arms}, flush=True)
    print('  mean gate per subtype:', cg, flush=True)


if __name__ == '__main__':
    main()
