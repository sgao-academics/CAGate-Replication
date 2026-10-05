# -*- coding: utf-8 -*-
"""STRING threshold sensitivity for the external-support claim.

Reviewer point R1.3 asks for the support rate at combined scores >=400, >=700 and
>=900, not >=700 alone.  This job rebuilds all three STRING pools in a single pass
(so the 154 MB PPI file is parsed once), then recomputes, per cancer and per arm,
the fraction of each arm's top-k edges that appear in the pool, k = the gate's own
edge count (the convention used in the manuscript).

Arms: gate / base / notears (from mega33_w/*.npz) and GENIE3 (from
b3_genie3/vim/*.npz, author's official RF implementation, ntrees=1000).

Null: every ordered pair (i != j) on the same 100-gene panel -- deterministic,
no sampling, so the null rate is a property of the panel and the threshold.

Outputs, all checkpointed per cancer:
  b3_cache/string_pairs_{400,700,900}.pkl
  b3_cache/ensp2sym.pkl
  b3_stringthr/<CANCER>.json
  BATCH3_THRESH.md
"""
import os, sys, json, gzip, pickle, time, glob
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
VDIR = os.environ.get('STRING_DATA_DIR', '')
WDIR = os.path.join(DATA_DIR, 'mega33_w')
VIMDIR = os.path.join(DATA_DIR, 'genie3', 'vim')
CACHE = os.environ.get('STRING_CACHE_DIR', os.path.join(DATA_DIR, '_cache'))
OUT = os.path.join(DATA_DIR, 'string_threshold')
THR_LIST = [400, 700, 900]
WTHR = 0.3
ARMS = ['genie3', 'gate', 'base', 'notears']

import numpy as np


# ── STRING pools ─────────────────────────────────────────────────────────────
def ensp2sym():
    cache = os.path.join(CACHE, 'ensp2sym.pkl')
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
    os.makedirs(CACHE, exist_ok=True)
    with open(cache, 'wb') as fh:
        pickle.dump(e2s, fh)
    print('  ensp2sym cached: %d accessions' % len(e2s), flush=True)
    return e2s


def build_pools():
    """single pass over the PPI file -> one directed-pair pool per threshold"""
    have = {}
    for t in THR_LIST:
        p = os.path.join(CACHE, 'string_pairs_%d.pkl' % t)
        if os.path.exists(p):
            with open(p, 'rb') as fh:
                have[t] = pickle.load(fh)
    missing = [t for t in THR_LIST if t not in have]
    legacy = os.path.join(CACHE, 'string_pairs.pkl')
    if 700 in missing and os.path.exists(legacy):
        with open(legacy, 'rb') as fh:
            have[700] = pickle.load(fh)
        missing = [t for t in THR_LIST if t not in have]
        print('  >=700 pool reused from legacy string_pairs.pkl (%d pairs)'
              % len(have[700]), flush=True)
    if missing:
        e2s = ensp2sym()
        pools = {t: set() for t in missing}
        t0 = time.time()
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
                if sc < min(missing):
                    continue
                sa = e2s.get(p[0].split('.', 1)[-1])
                sb = e2s.get(p[1].split('.', 1)[-1])
                if not sa or not sb or sa == sb:
                    continue
                for t in missing:
                    if sc >= t:
                        pools[t].add((sa, sb))
                        pools[t].add((sb, sa))
        os.makedirs(CACHE, exist_ok=True)
        for t in missing:
            with open(os.path.join(CACHE, 'string_pairs_%d.pkl' % t), 'wb') as fh:
                pickle.dump(pools[t], fh)
            have[t] = pools[t]
        print('  pools built in %.0fs' % (time.time() - t0), flush=True)
    for t in THR_LIST:
        print('  >=%d pool: %d directed pairs' % (t, len(have[t])), flush=True)
    return have


# ── per-cancer work ──────────────────────────────────────────────────────────
def topk(score, k):
    s = np.abs(np.asarray(score, dtype=float).copy())
    np.fill_diagonal(s, 0.0)
    d = s.shape[0]
    flat = np.argsort(-s.ravel())[:k]
    return [(int(f) // d, int(f) % d) for f in flat]


def natset(score):
    """the arm's own edge set at |W| > 0.3 -- the convention of the main text"""
    s = np.abs(np.asarray(score, dtype=float).copy())
    np.fill_diagonal(s, 0.0)
    ii, jj = np.where(s > WTHR)
    return [(int(i), int(j)) for i, j in zip(ii, jj)]


def supp(genes, pairs, pool):
    if not pairs:
        return 0.0
    return round(100.0 * sum(1 for i, j in pairs if (genes[i], genes[j]) in pool)
                 / len(pairs), 2)


def unit(cancer, pools):
    fp = os.path.join(OUT, '%s.json' % cancer)
    if os.path.exists(fp):
        return 'skip'
    js = os.path.join(WDIR, '%s.json' % cancer)
    zz = os.path.join(WDIR, '%s.npz' % cancer)
    if not (os.path.exists(js) and os.path.exists(zz)):
        return 'miss'
    meta = json.load(open(js, encoding='utf-8'))
    genes = [g.upper() for g in meta['genes']]
    z = np.load(zz, allow_pickle=True)
    W = {'gate': z['gate'], 'base': z['base'], 'notears': z['notears']}
    vp = os.path.join(VIMDIR, '%s.npz' % cancer)
    if os.path.exists(vp):
        W['genie3'] = np.load(vp, allow_pickle=True)['VIM']
    d = len(genes)
    # gate edge count excluding the diagonal (mirrors _b3_genie3.n_edges)
    Wg = np.abs(W['gate'].copy())
    np.fill_diagonal(Wg, 0.0)
    kg = int((Wg > WTHR).sum())
    all_pairs = [(i, j) for i in range(d) for j in range(d) if i != j]
    n = meta.get('n')
    if n is None:
        n = int(np.load(os.path.join(CACHE, '%s_d100.npz' % cancer),
                        allow_pickle=True)['X'].shape[0]) \
            if os.path.exists(os.path.join(CACHE, '%s_d100.npz' % cancer)) else None
    rec = {'cancer': cancer, 'n': int(n) if n is not None else None,
           'd': d, 'k_gate': kg, 'arms': {}}
    for a in ARMS:
        if a not in W:
            continue
        km = topk(W[a], kg)          # count-matched: this arm's top-k by |score|
        own = natset(W[a])           # this arm's own |W| > 0.3 edge set
        rec['arms'][a] = {'edges_nat': len(own), 'edges_topk': len(km)}
        for t in THR_LIST:
            pool = pools[t]
            rec['arms'][a]['gt%d' % t] = supp(genes, km, pool)       # matched
            rec['arms'][a]['nat%d' % t] = supp(genes, own, pool)     # own set
    rec['random'] = {}
    for t in THR_LIST:
        pool = pools[t]
        rec['random']['gt%d' % t] = round(
            100.0 * sum(1 for i, j in all_pairs if (genes[i], genes[j]) in pool)
            / len(all_pairs), 2)
    os.makedirs(OUT, exist_ok=True)
    tmp = fp + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(rec, fh, indent=2)
    os.replace(tmp, fp)
    print('  %-5s n=%s k=%d  ' % (cancer, rec['n'], kg) +
          '  '.join('%s[%s]' % (a, '/'.join('%.1f' % rec['arms'][a]['gt%d' % t]
                                            for t in THR_LIST))
                    for a in [x for x in ARMS if x in rec['arms']]), flush=True)
    return 'ok'


# ── summary ──────────────────────────────────────────────────────────────────
def wilcoxon_p(a, b):
    try:
        from scipy.stats import wilcoxon
        if np.all(np.asarray(a) == np.asarray(b)):
            return 1.0
        return float(wilcoxon(a, b).pvalue)
    except Exception:
        return float('nan')


def summarise():
    rows = []
    for fp in sorted(glob.glob(os.path.join(OUT, '*.json'))):
        rows.append(json.load(open(fp, encoding='utf-8')))
    if not rows:
        print('no results')
        return
    def table(prefix, title, note):
        out = ['', '#### %s' % title, '', note, '',
               '| threshold | arm | mean edges | mean support % | median | vs random (x) |',
               '|:--|:--|--:|--:|--:|--:|']
        for t in THR_LIST:
            key = prefix + str(t)
            rnd = np.mean([r['random']['gt%d' % t] for r in rows])
            for a in ARMS:
                sub = [r for r in rows if a in r['arms']]
                if not sub:
                    continue
                v = [r['arms'][a][key] for r in sub]
                ne = np.mean([r['arms'][a]['edges_nat' if prefix == 'nat' else 'edges_topk']
                              for r in sub])
                if prefix == 'nat' and ne == 0:
                    continue          # GENIE3 has no |W| scale: no natural edge set
                out.append('| >=%d | %s | %.1f | %.2f | %.2f | %.1f |'
                           % (t, a, ne, np.mean(v), np.median(v),
                              np.mean(v) / rnd if rnd else float('nan')))
            out.append('| >=%d | *random* | - | %.2f | - | 1.0 |' % (t, rnd))
        out += ['', '| threshold | comparison | mean diff (pp) | cohorts positive | Wilcoxon P |',
                '|:--|:--|--:|:--|--:|']
        for t in THR_LIST:
            key = prefix + str(t)
            for a, b in (('gate', 'notears'), ('gate', 'base'), ('base', 'notears')):
                va = np.array([r['arms'][a][key] for r in rows if a in r['arms']])
                vb = np.array([r['arms'][b][key] for r in rows if b in r['arms']])
                if len(va) == 0 or len(vb) == 0:
                    continue
                diff = va - vb
                out.append('| >=%d | %s - %s | %+.2f | %d/%d | %.3g |'
                           % (t, a, b, diff.mean(), int((diff > 0).sum()), len(diff),
                              wilcoxon_p(va, vb)))
        return out

    L = ['## STRING threshold sensitivity (>=400 / >=700 / >=900)', '',
         'Fraction of a method\'s edges whose protein pair appears in STRING v12 at '
         'the given combined score. The null is every ordered pair (i != j) on the '
         'same 100-gene panel, so it carries no sampling error. Two conventions are '
         'reported separately, because they measure different quantities: the '
         '**natural** set is each arm\'s own edges at |W| > %g (the convention of the '
         'main text), the **count-matched** set is each arm\'s top-k by |score| with '
         'k = that cohort\'s gate edge count.' % WTHR]
    L += table('nat', 'A. Natural edge set, |W| > %g (main-text convention)' % WTHR,
               'The gate\'s headline statistic; GENIE3 has no |W| scale and so is absent here.')
    L += table('gt', 'B. Count-matched top-k (defensive statistic)',
               'Equal edge budgets; this is the comparison that returns a null result '
               'between the gated and ungated arms.')
    L.append('')
    with open(os.path.join(C, 'BATCH3_THRESH.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L))
    print('\n'.join(L))
    print('\n-> b3_stringthr/*.json  BATCH3_THRESH.md')


def main():
    t0 = time.time()
    print('JOB D / STRING threshold sensitivity', flush=True)
    os.makedirs(OUT, exist_ok=True)
    pools = build_pools()
    cases = [os.path.basename(p)[:-5] for p in sorted(glob.glob(os.path.join(WDIR, '*.json')))]
    print('  cancers: %d' % len(cases), flush=True)
    for c in cases:
        try:
            unit(c, pools)
        except Exception as e:
            print('  %-5s ERROR %s %s' % (c, type(e).__name__, e), flush=True)
    summarise()
    print('JOB D done (%.0fs)' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
