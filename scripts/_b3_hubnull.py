# -*- coding: utf-8 -*-
"""Is GENIE3's real-data STRING advantage real, or a hub/degree artefact?

Random-forest importances concentrate on a few regulators; hubs are dense in
STRING.  So for every arm we take the SAME number of top edges (k = CAGate gate
count) and compare its STRING support against a DEGREE-MATCHED null: we keep the
observed in-/out-degree sequences of the selected edge set and re-pair the stubs
at random (stub matching).  If an arm's support stops exceeding its own matched
null, its "advantage" is just the degree profile it happened to pick.

Reports per arm: observed %, matched-null %, z, empirical p, distinct genes,
max out-degree.
"""
import os, sys, json, glob, pickle
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Paths are resolved relative to this file so the script runs from a fresh
# clone. External inputs (TCGA panels, STRING v12 raw files) are located via
# the environment variables documented in the repository README.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
C = SCRIPTS_DIR
WDIR = os.path.join(DATA_DIR, 'mega33_w')
VIMDIR = os.path.join(DATA_DIR, 'genie3', 'vim')
THR = 0.3
R = 300
import numpy as np
from collections import Counter

ARMS = ['genie3', 'gate', 'base', 'notears']


def topk_edges(score, k):
    s = np.abs(score.copy())
    np.fill_diagonal(s, 0.0)
    flat = np.argsort(-s.ravel())[:k]
    d = score.shape[0]
    return [(int(f) // d, int(f) % d) for f in flat]


def support_pct(G, pairs, pool):
    if not pairs:
        return 0.0
    return 100.0 * sum(1 for i, j in pairs if (G[i], G[j]) in pool) / len(pairs)


def matched_null(G, pairs, pool, rng):
    out_stubs = [i for i, j in pairs]
    in_stubs = [j for i, j in pairs]
    vals = []
    for _ in range(R):
        perm = rng.permutation(len(in_stubs))
        v = 0
        n = 0
        for a, b in zip(out_stubs, perm):
            t = in_stubs[b]
            if a == t:
                continue
            n += 1
            if (G[a], G[t]) in pool:
                v += 1
        vals.append(100.0 * v / max(n, 1))
    return float(np.mean(vals)), float(np.std(vals))


def main():
    with open(os.path.join(C, 'b3_cache', 'string_pairs.pkl'), 'rb') as fh:
        pool = pickle.load(fh)
    rng = np.random.default_rng(0)
    rows = []
    for js in sorted(glob.glob(os.path.join(WDIR, '*.json'))):
        c = os.path.basename(js)[:-5]
        vp = os.path.join(VIMDIR, '%s.npz' % c)
        zz = os.path.join(WDIR, '%s.npz' % c)
        if not (os.path.exists(vp) and os.path.exists(zz)):
            print('  %-5s: missing assets, skip' % c, flush=True)
            continue
        genes = json.load(open(js, encoding='utf-8'))['genes']
        G = [g.upper() for g in genes]
        vim = np.load(vp, allow_pickle=True)['VIM']
        z = np.load(zz, allow_pickle=True)
        k = int((np.abs(z['gate']) > THR).sum())
        rec = {'cancer': c, 'k': k, 'arms': {}}
        for a, S in (('genie3', vim), ('gate', z['gate']), ('base', z['base']),
                     ('notears', z['notears'])):
            ed = topk_edges(S, k)
            obs = support_pct(G, ed, pool)
            mu, sd = matched_null(G, ed, pool, rng)
            zsc = (obs - mu) / sd if sd > 1e-9 else float('nan')
            od = Counter(i for i, j in ed)
            rec['arms'][a] = {'obs': round(obs, 2), 'null': round(mu, 2),
                              'z': round(zsc, 2), 'n_genes': len(set(
                                  [i for i, j in ed] + [j for i, j in ed])),
                              'max_outdeg': max(od.values()) if od else 0}
        rows.append(rec)
        print('  %-5s k=%-3d ' % (c, k) + ' | '.join(
            '%s obs=%.1f%% null=%.1f%% z=%+.2f' % (a, rec['arms'][a]['obs'],
            rec['arms'][a]['null'], rec['arms'][a]['z']) for a in ARMS), flush=True)
    out = os.path.join(C, 'b3_hubnull.json')
    json.dump(rows, open(out, 'w', encoding='utf-8'), indent=2)

    L = ['## Degree-matched null on the real-data STRING endpoint', '',
         'For each arm: top-k edges (k = CAGate gate count), observed STRING>=700 '
         'support vs a stub-matched null that preserves the edge set in/out degrees.', '',
         '| arm | observed % | matched-null % | mean z | distinct genes | max out-deg |',
         '|:--|--:|--:|--:|--:|--:|']
    for a in ARMS:
        o = np.mean([r['arms'][a]['obs'] for r in rows])
        n = np.mean([r['arms'][a]['null'] for r in rows])
        z = np.nanmean([r['arms'][a]['z'] for r in rows])
        g = np.mean([r['arms'][a]['n_genes'] for r in rows])
        md = np.mean([r['arms'][a]['max_outdeg'] for r in rows])
        L.append('| %s | %.2f | %.2f | %+.2f | %.1f | %.1f |' % (a, o, n, z, g, md))
    L.append('')
    L.append('read: an arm whose observed% ~ its matched-null% (z ~ 0) has no STRING '
             'advantage beyond the degree profile it selected.')
    open(os.path.join(C, 'BATCH3_HUBNULL.md'), 'w', encoding='utf-8').write('\n'.join(L))
    print('\n'.join(L))
    print('\n-> b3_hubnull.json / BATCH3_HUBNULL.md')


if __name__ == '__main__':
    main()
