# -*- coding: utf-8 -*-
"""A-M4: external support restricted to STRING's *physical* evidence channels.

Motivation (reviewer M4): the combined STRING score mixes co-expression and
text-mining channels with experimental/database evidence.  Because the arms'
inputs are themselves expression-derived, validating with the combined score is
partly circular.  This script rebuilds the external-support statistic using only
the ``experiments`` and ``database`` channels (their transferred variants, and
the co-expression / text-mining channels, are excluded), and recomputes the
paired per-cohort comparison gate vs NOTEARS vs the ungated arm.

An edge is counted as *physically* supported when its STRING ``experiments`` or
``database`` channel score is >= 700 -- the same numeric cut that is applied to
the combined score elsewhere in the manuscript.  The combined score itself is a
probabilistic fusion of all channels and is not re-derived here.

Reads only: mega33_w (the three arms' weight matrices, one npz + json per
cancer), STRING v12 flat files.  Writes A_M4_phys.json next to this script.
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
import os, sys, glob, json, gzip
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from scipy.stats import wilcoxon

CFIX = DATA_DIR
WDIR = os.path.join(CFIX, 'mega33_w')
VDIR = STRING_DATA_DIR
OUT = os.path.join(CFIX, 'A_M4_phys.json')

THRS = [0.3, 0.7]       # 0.3 = main convention, 0.7 = the strict cut of SN23
COMBINED_CUT = 700
CHANNEL_CUT = 700       # experiments / database channel cut
N_RAND = 100000         # random ordered pairs sampled per cohort
SEED = 20261005


def log(*a):
    print(*a, flush=True)


def main():
    # ── gene panels ─────────────────────────────────────────────────────────
    meta = {}
    allg = set()
    for f in sorted(glob.glob(os.path.join(WDIR, '*.json'))):
        d = json.load(open(f, encoding='utf-8'))
        if d.get('status') != 'OK':
            continue
        meta[d['cancer']] = d
        allg.update(g.upper() for g in d['genes'])
    log('[genes] %d cancers, %d symbols' % (len(meta), len(allg)))

    # ── STRING symbol <-> ENSP ──────────────────────────────────────────────
    sym2ensp = defaultdict(set)
    with gzip.open(os.path.join(VDIR, 'string_info.txt.gz'), 'rt',
                   encoding='utf-8', errors='ignore') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) >= 2:
                sym2ensp[p[1].upper()].add(p[0].split('.', 1)[-1])
    ensp2sym = {}
    for s, es in sym2ensp.items():
        for e in es:
            ensp2sym.setdefault(e, s)
    our_ensp = set()
    for g in allg:
        our_ensp |= sym2ensp.get(g, set())
    log('[STRING] our ENSP pool: %d' % len(our_ensp))

    # ── STRING pairs: combined>=700  AND  physical (exp|db channel >=700) ───
    comb_pairs = set()
    phys_pairs = set()
    with gzip.open(os.path.join(VDIR, 'string_ppi_full.txt.gz'), 'rt',
                   encoding='utf-8', errors='ignore') as f:
        header = f.readline().rstrip('\n')
        cols = header.split()
        idx_exp = next((k for k, c in enumerate(cols) if c == 'experiments'), None)
        idx_db = next((k for k, c in enumerate(cols) if c == 'database'), None)
        log('[STRING] cols=%d exp=%s db=%s' % (len(cols), idx_exp, idx_db))
        n = 0
        for line in f:
            n += 1
            p = line.rstrip('\n').split()
            if len(p) < 3:
                continue
            try:
                sc = int(p[-1])
            except ValueError:
                continue
            a = p[0].split('.', 1)[-1]
            b = p[1].split('.', 1)[-1]
            if a not in our_ensp or b not in our_ensp:
                continue
            sa = ensp2sym.get(a)
            sb = ensp2sym.get(b)
            if not (sa and sb and sa != sb):
                continue
            if sc >= COMBINED_CUT:
                comb_pairs.add((sa, sb))
                comb_pairs.add((sb, sa))
            if idx_exp is not None and idx_db is not None and len(p) > max(idx_exp, idx_db):
                try:
                    pe = int(p[idx_exp])
                    pd = int(p[idx_db])
                except ValueError:
                    pe = pd = 0
                if pe >= CHANNEL_CUT or pd >= CHANNEL_CUT:
                    phys_pairs.add((sa, sb))
                    phys_pairs.add((sb, sa))
        log('[STRING] scanned %d lines | combined>=700 dir-pairs %d | physical dir-pairs %d'
            % (n, len(comb_pairs), len(phys_pairs)))

    # ── per-cohort, per-arm support ─────────────────────────────────────────
    rng = np.random.default_rng(SEED)
    per = defaultdict(lambda: defaultdict(dict))   # cancer -> thr -> arm
    arms = ('notears', 'base', 'gate')
    for c, d in sorted(meta.items()):
        npz = os.path.join(WDIR, '%s.npz' % c)
        if not os.path.exists(npz):
            continue
        z = np.load(npz)
        G = [g.upper() for g in d['genes']]
        dd = len(G)
        # zero model: an edge drawn at random from this panel; the physical
        # support rate is a per-edge probability, independent of the budget.
        ri = rng.integers(0, dd, size=N_RAND)
        rj = rng.integers(0, dd, size=N_RAND)
        rh = sum(1 for i, j in zip(ri, rj)
                 if i != j and (G[i], G[j]) in phys_pairs)
        rand_rate = rh / N_RAND
        for t in THRS:
            for arm in arms:
                W = z[arm]
                idx = [(i, j) for i in range(dd) for j in range(dd)
                       if i != j and abs(W[i, j]) > t]
                n = len(idx)
                hc = sum(1 for i, j in idx if (G[i], G[j]) in comb_pairs)
                hp = sum(1 for i, j in idx if (G[i], G[j]) in phys_pairs)
                per[c][t][arm] = dict(
                    n=n, comb=hc, phys=hp,
                    comb_rate=(hc / n if n else 0.0),
                    phys_rate=(hp / n if n else 0.0),
                    phys_random=rand_rate)
        log('  %-6s d=%d done' % (c, dd))

    # ── paired tests ────────────────────────────────────────────────────────
    def paired(t, a, b):
        xs = [(float(per[c][t][a]['phys_rate']), float(per[c][t][b]['phys_rate']))
              for c in per if a in per[c][t] and b in per[c][t]]
        diffs = np.array([x - y for x, y in xs]) * 100
        try:
            st, pv = wilcoxon(diffs)
        except Exception:
            st, pv = float('nan'), float('nan')
        return dict(mean_diff=float(diffs.mean()), median_diff=float(np.median(diffs)),
                    n_win=int((diffs > 0).sum()), n_tie=int((diffs == 0).sum()),
                    n=len(diffs), p=float(pv))

    summary = {}
    log('\n### A-M4 physical support (exp/db channel >=700; co-expr & text-mining excluded)')
    for t in THRS:
        rows = {}
        for arm in ('gate', 'base', 'notears'):
            r = np.mean([per[c][t][arm]['phys_rate'] for c in per]) * 100
            rc = np.mean([per[c][t][arm]['comb_rate'] for c in per]) * 100
            rr = np.mean([per[c][t][arm]['phys_random'] for c in per]) * 100
            rows[arm] = dict(phys_mean=r, comb_mean=rc, phys_random_mean=rr)
            log('  |W|>%.1f  %-8s physical %.2f%%   (combined %.2f%%, random %.2f%%)'
                % (t, arm, r, rc, rr))
        gv = paired(t, 'gate', 'notears')
        gu = paired(t, 'gate', 'base')
        log('    gate-vs-NOTEARS  %+.2f pp  (%d/%d cohorts >0, P=%.3g)'
            % (gv['mean_diff'], gv['n_win'], gv['n'], gv['p']))
        log('    gate-vs-ungated  %+.2f pp  (%d/%d cohorts >0, P=%.3g)'
            % (gu['mean_diff'], gu['n_win'], gu['n'], gu['p']))
        summary[str(t)] = dict(arms=rows, gate_vs_notears=gv, gate_vs_base=gu)

    json.dump(dict(summary=summary,
                   per={c: {str(t): per[c][t] for t in per[c]} for c in per},
                   n_comb_pairs=len(comb_pairs), n_phys_pairs=len(phys_pairs),
                   channel_cut=CHANNEL_CUT, combined_cut=COMBINED_CUT, seed=SEED),
              open(OUT, 'w', encoding='utf-8'), indent=1, default=str)
    log('\nsaved -> %s' % OUT)


if __name__ == '__main__':
    main()
