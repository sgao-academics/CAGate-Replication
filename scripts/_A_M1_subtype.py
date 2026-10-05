# -*- coding: utf-8 -*-
"""A-M1: do the K-means clusters track known biology, and does the gate need them to?

Reviewer M1 asks for the cluster labels to be back-annotated against molecular
subtype / immune / purity structure, and for the gate to be re-run with the
*true* labels.  This script reports, for TCGA-BRCA (the cohort used throughout):

  1. ARI between the K-means partition (KMeans K=number of published subtypes,
     random_state=0, n_init=10 -- the pipeline setting) and the published PAM50
     call, with the full contingency table.
  2. The per-group residual dispersion sigma_g of the fitted base graph, for the
     K-means partition and for the PAM50 partition -- the quantity the gate
     reads -- so the two can be compared on the same footing.
  3. Expression-derived scores for immune infiltration, stromal content and
     proliferation, per K-means cluster and per PAM50 subtype (Kruskal-Wallis),
     and their correlation with the per-cluster sigma.

Everything is read from existing artefacts; no new solver run is made here.
Writes A_M1_subtype.json next to this script.
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
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.stats import kruskal, spearmanr

DATA = TCGA_DATA_DIR
SUB = os.path.join(DATA_DIR, 'subtypes')
RUN = os.path.join(DATA_DIR, 'subtypes_run')
OUT = os.path.join(DATA_DIR, 'A_M1_subtype.json')

THR = 0.3
MARKERS = {
    'immune': ['PTPRC', 'CD3D', 'CD3E', 'CD2', 'CD8A', 'CD19', 'MS4A1', 'NKG7',
               'GZMB', 'PRF1', 'CD68', 'LYZ', 'IL2RG', 'CXCL9', 'CXCL10'],
    'stromal': ['COL1A1', 'COL1A2', 'COL3A1', 'FAP', 'PDGFRB', 'ACTA2', 'THY1',
                'DCN', 'LUM', 'COL5A1', 'COL6A1'],
    'proliferation': ['MKI67', 'TOP2A', 'PCNA', 'CCNB1', 'BIRC5', 'AURKA', 'PLK1',
                      'CCNA2', 'UBE2C', 'BUB1'],
}


def log(*a):
    print(*a, flush=True)


def main():
    # ── panel + ids, exactly as the pipeline builds them ────────────────────
    df = pd.read_csv(os.path.join(DATA, 'TCGA_BRCA_HiSeqV2.tsv'), sep='\t', index_col=0)
    genes_all = list(df.index)
    data = df.values.astype(np.float64)
    top = np.argsort(data.var(axis=1))[-100:]
    genes = [genes_all[i] for i in top]
    X = data[top, :].T
    X = (X - X.mean(0)) / np.clip(X.std(0), 1e-6, None)
    ids = list(df.columns)
    log('[panel] %d genes, %d samples' % (len(genes), len(ids)))

    # ── PAM50 ───────────────────────────────────────────────────────────────
    def norm(s):
        m = re.match(r'(TCGA-[0-9A-Za-z]{2}-[0-9A-Za-z]{4})', str(s).strip())
        return m.group(1) if m else str(s)[:15]

    clin = pd.read_csv(os.path.join(SUB, 'BRCA_clinical.tsv'), sep='\t', low_memory=False)
    lab = clin[[clin.columns[0], 'PAM50Call_RNAseq']].copy()
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
    log('[PAM50] %d samples, %d subtypes %s, sizes %s'
        % (len(idx), len(cats), cats, np.bincount(cid_p).tolist()))

    # ── K-means, the pipeline setting ───────────────────────────────────────
    cid_km = KMeans(n_clusters=len(cats), random_state=0, n_init=10).fit_predict(Xs)
    ari = adjusted_rand_score(cid_p, cid_km)
    log('[ARI] K-means vs PAM50 = %.3f' % ari)
    K = len(cats)
    cont = np.zeros((K, K), dtype=int)
    for a, b in zip(cid_p, cid_km):
        cont[a, b] += 1
    log('contingency (rows PAM50 %s, cols K-means):' % cats)
    for i, c in enumerate(cats):
        log('  %-8s %s' % (c, cont[i].tolist()))

    # ── residual dispersion sigma_g under the fitted base graph ────────────
    z = np.load(os.path.join(RUN, 'BRCA.npz'))
    W = z['base']
    R = Xs - Xs @ W
    res = (R ** 2).mean(axis=1)

    def sigmas(cid):
        out = []
        for g in range(int(cid.max()) + 1):
            v = res[cid == g]
            out.append(float(v.std(ddof=1)) if v.size > 1 else 0.0)
        return np.array(out)

    s_p, s_km = sigmas(cid_p), sigmas(cid_km)
    log('\n[dispersion] PAM50 sigma: %s  spread=%.5f rel=%.3f'
        % (np.round(s_p, 5).tolist(), s_p.max() - s_p.min(),
           (s_p.max() - s_p.min()) / np.median(s_p)))
    log('[dispersion] K-means sigma: %s  spread=%.5f rel=%.3f'
        % (np.round(s_km, 5).tolist(), s_km.max() - s_km.min(),
           (s_km.max() - s_km.min()) / np.median(s_km)))

    # ── expression-derived biological scores ───────────────────────────────
    gi = {g.upper(): i for i, g in enumerate(genes_all)}
    scores = {}
    for name, mk in MARKERS.items():
        rows = [gi[g] for g in mk if g in gi]
        log('[score] %-14s %d/%d markers found' % (name, len(rows), len(mk)))
        v = data[rows, :].mean(axis=0)
        v = (v - v.mean()) / (v.std() + 1e-12)
        scores[name] = v[idx]
    # purity proxy: high immune+stroma => low tumour content
    scores['purity_proxy'] = -(scores['immune'] + scores['stromal'])

    log('\n[per-cluster mean score]')
    out_scores = {}
    for name, v in scores.items():
        kw_p = kruskal(*[v[cid_km == g] for g in range(K)])[1] if K > 1 else 1.0
        kw_pp = kruskal(*[v[cid_p == g] for g in range(K)])[1] if K > 1 else 1.0
        km_means = [round(float(v[cid_km == g].mean()), 3) for g in range(K)]
        p_means = [round(float(v[cid_p == g].mean()), 3) for g in range(K)]
        rho, prho = spearmanr(s_km, [v[cid_km == g].mean() for g in range(K)])
        out_scores[name] = dict(km_means=km_means, pam50_means=p_means,
                                kruskal_km_p=float(kw_p), kruskal_pam50_p=float(kw_pp),
                                rho_with_sigma=float(rho), p_rho=float(prho))
        log('  %-14s K-means p=%.4f %s | PAM50 p=%.4f %s | rho(sigma)=%.2f (p=%.3f)'
            % (name, kw_p, km_means, kw_pp, p_means, rho, prho))

    json.dump(dict(ari=float(ari), pam50_cats=cats,
                   contingency=cont.tolist(),
                   km_sizes=np.bincount(cid_km).tolist(),
                   pam50_sizes=np.bincount(cid_p).tolist(),
                   pam50_sigma=s_p.tolist(), km_sigma=s_km.tolist(),
                   scores=out_scores, n=len(idx)),
              open(OUT, 'w', encoding='utf-8'), indent=1)
    log('\nsaved -> %s' % OUT)


if __name__ == '__main__':
    main()
