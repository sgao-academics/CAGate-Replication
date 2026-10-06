# -*- coding: utf-8 -*-
"""Cross-cancer consensus pipeline, step 2: recurrence ladder and seed stability.

Rebuilds the recurrence ladder of Table S7 from two directories of per-cohort
weight matrices (step 1), and reports how far the ladder and the consensus set
move when only the subsample seed changes (Note SN28).

Counts and overlap statistics are self-contained.  The external-support columns
need the derived source files, whose locations come from the environment:

  STRING_SUPPORT_JSON   JSON with "STRING" and "BioGRID" pair lists
                        (derived from STRING v12 and BioGRID for this panel)
  HALLMARK_JSON         JSON with a "pairs" list (MSigDB Hallmark co-membership)

Usage:
  python scripts/_A_M1e_consensus_seed_stability.py data/consensus_w_s0 data/consensus_w_s1
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THR = 0.3
BANDS = [('1', 1, 1), ('2-3', 2, 3), ('4-7', 4, 7), ('8-15', 8, 15), ('>=16', 16, 10 ** 9)]


def _pairs(path, key):
    if not path or not os.path.exists(path):
        return None
    o = json.load(open(path, encoding='utf-8'))
    return {(min(a, b), max(a, b)) for a, b in o[key]}


STRG = _pairs(os.environ.get('STRING_SUPPORT_JSON', ''), 'STRING')
BIO = _pairs(os.environ.get('STRING_SUPPORT_JSON', ''), 'BioGRID')
HAL = _pairs(os.environ.get('HALLMARK_JSON', ''), 'pairs')


def export(wdir):
    import glob
    fs = sorted(glob.glob(os.path.join(wdir, '*.npz')))
    assert fs, 'no npz in %s' % wdir
    genes, Ws = None, []
    for f in fs:
        z = np.load(f, allow_pickle=True)
        genes = [str(g).upper() for g in z['genes']]
        Ws.append(z['W'])
    d = len(genes)
    cnt = np.zeros((d, d), np.int16)
    for w in Ws:
        cnt += (np.abs(w) > THR).astype(np.int16)
    np.fill_diagonal(cnt, 0)
    E = {(i, j): int(cnt[i, j]) for i in range(d) for j in range(d)
         if i != j and cnt[i, j] > 0}
    return genes, E


def ladder(genes, E):
    rows = []
    for name, lo, hi in BANDS:
        ks = [(i, j) for (i, j), c in E.items() if lo <= c <= hi]
        if not ks:
            rows.append(dict(band=name, n=0))
            continue
        pairs = [(genes[i], genes[j]) for i, j in ks]
        row = dict(band=name, n=len(ks))
        for lbl, s in (('string', STRG), ('biogrid', BIO), ('hallmark', HAL)):
            if s is not None:
                row[lbl] = round(100 * float(np.mean(
                    [(min(a, b), max(a, b)) in s for a, b in pairs])), 1)
        rows.append(row)
    return dict(n_pairs=len(E), consensus_ge4=sum(1 for c in E.values() if c >= 4), bands=rows)


def main(a, b):
    ga, Ea = export(a)
    gb, Eb = export(b)
    assert ga == gb, 'panels differ between the two runs'

    la, lb = ladder(ga, Ea), ladder(gb, Eb)
    print('=== ladder: %s (published seed) vs %s (second seed) ===' % (a, b))
    print('  %-6s %8s %12s %8s %12s' % ('band', 'n (A)', 'STRING (A)', 'n (B)', 'STRING (B)'))
    for ra, rb in zip(la['bands'], lb['bands']):
        print('  %-6s %8d %12s %8d %12s' % (ra['band'], ra['n'], ra.get('string', '-'),
                                            rb['n'], rb.get('string', '-')))
    print('  consensus (cnt>=4): %d vs %d' % (la['consensus_ge4'], lb['consensus_ge4']))

    sa, sb = set(Ea), set(Eb)
    inter, uni = sa & sb, sa | sb
    cc = np.array([Ea[k] for k in inter]); dd = np.array([Eb[k] for k in inter])
    from scipy.stats import spearmanr
    rho, _ = spearmanr(cc, dd)
    Ca = {k for k, c in Ea.items() if c >= 4}
    Cb = {k for k, c in Eb.items() if c >= 4}
    ov = dict(
        jaccard_obs=len(inter) / len(uni),
        recall_A_in_B=len(inter) / len(sa), recall_B_in_A=len(inter) / len(sb),
        n_shared=len(inter), max_absdiff=int(np.abs(cc - dd).max()),
        frac_exact=float(np.mean(cc == dd)), frac_absdiff_le1=float(np.mean(np.abs(cc - dd) <= 1)),
        spearman_shared=float(rho),
        consensus_A=len(Ca), consensus_B=len(Cb), consensus_shared=len(Ca & Cb),
        consensus_jaccard=len(Ca & Cb) / len(Ca | Cb))
    print('  observed pairs %d vs %d | Jaccard %.3f | Spearman %.3f | consensus %d vs %d (Jaccard %.3f)'
          % (len(sa), len(sb), ov['jaccard_obs'], ov['spearman_shared'],
             len(Ca), len(Cb), ov['consensus_jaccard']))

    out = os.path.join(REPO, 'data')
    os.makedirs(out, exist_ok=True)
    # Record the two input directories by their last path component only, so that
    # no absolute path from the machine that ran the analysis is written to disk.
    json.dump(dict(ladder_published=la, ladder_second_seed=lb,
                   overlap=ov, dirs=[os.path.basename(os.path.normpath(p)) for p in (a, b)]),
              open(os.path.join(out, 'A_M1_seedstab.json'), 'w', encoding='utf-8'), indent=1)
    print('saved -> data/A_M1_seedstab.json')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
