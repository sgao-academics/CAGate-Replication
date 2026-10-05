# -*- coding: utf-8 -*-
"""A-M4b: the physical-channel control, re-run at a MATCHED edge count.

Reviewer A-M1: Note SN24 reports the gate ahead of NOTEARS by +1.17 pp on the
forensic ``experiments``/``database`` channels -- but on each arm's own natural
edge set (gate ~50 edges vs NOTEARS ~64).  The volume-matched comparison of the
main text (Section 4.2) fixes the count by truncating the baselines to the
gate's edge budget by |W|; the same must be done for the physical channels, or
the +1.17 pp could just be the count effect that the main text rules out.

This script repeats SN24 on a matched count: for each cohort, both ungated arms
are truncated to the gate's number of edges at the same threshold, ranked by
|W|, and the paired comparison is redone on the physical support rate.  For
validation, the same truncation is also applied to the combined score, which
should reproduce the released count-matched null of Section 4.2.

Reads only: mega33_w (three arms' weight matrices), STRING v12 flat files.
Writes A_M4b_phys_cm.json next to the other data files.
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
DATA_DIR = os.path.join(_REPO, 'data')
STRING_DATA_DIR = os.environ.get('STRING_DATA_DIR', '')

sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)
import glob, json, gzip
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from scipy.stats import wilcoxon

WDIR = os.path.join(DATA_DIR, 'mega33_w')
VDIR = STRING_DATA_DIR
OUT = os.path.join(DATA_DIR, 'A_M4b_phys_cm.json')

THRS = [0.3, 0.7]
COMBINED_CUT = 700
CHANNEL_CUT = 700


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

    comb_pairs = set()
    phys_pairs = set()
    with gzip.open(os.path.join(VDIR, 'string_ppi_full.txt.gz'), 'rt',
                   encoding='utf-8', errors='ignore') as f:
        header = f.readline().rstrip('\n')
        cols = header.split()
        idx_exp = next((k for k, c in enumerate(cols) if c == 'experiments'), None)
        idx_db = next((k for k, c in enumerate(cols) if c == 'database'), None)
        log('[STRING] cols=%d exp=%s db=%s' % (len(cols), idx_exp, idx_db))
        for line in f:
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
    log('[STRING] combined>=700 dir-pairs %d | physical dir-pairs %d'
        % (len(comb_pairs), len(phys_pairs)))

    # ── per cohort: gate own set, and the two baselines truncated to it ─────
    arms = ('gate', 'base', 'notears')
    per = defaultdict(lambda: defaultdict(dict))   # cancer -> thr -> arm
    n_gate_budget = defaultdict(dict)
    for c, d in sorted(meta.items()):
        npz = os.path.join(WDIR, '%s.npz' % c)
        if not os.path.exists(npz):
            continue
        z = np.load(npz)
        G = [g.upper() for g in d['genes']]
        dd = len(G)
        for t in THRS:
            gate_idx = [(i, j) for i in range(dd) for j in range(dd)
                        if i != j and abs(z['gate'][i, j]) > t]
            n_gate_budget[c][t] = len(gate_idx)
            for arm in arms:
                W = z[arm]
                all_idx = [(i, j) for i in range(dd) for j in range(dd)
                           if i != j and abs(W[i, j]) > t]
                if arm == 'gate':
                    idx = all_idx
                else:
                    all_idx.sort(key=lambda e: abs(W[e[0], e[1]]), reverse=True)
                    idx = all_idx[:len(gate_idx)]
                n = len(idx)
                hc = sum(1 for i, j in idx if (G[i], G[j]) in comb_pairs)
                hp = sum(1 for i, j in idx if (G[i], G[j]) in phys_pairs)
                per[c][t][arm] = dict(
                    n=n, comb=hc, phys=hp,
                    comb_rate=(hc / n if n else 0.0),
                    phys_rate=(hp / n if n else 0.0))
        log('  %-6s d=%d done (gate budget @0.3 = %d)'
            % (c, dd, n_gate_budget[c][0.3]))

    def paired(t, a, b, key):
        xs = [(float(per[c][t][a][key]), float(per[c][t][b][key]))
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
    log('\n### A-M4b physical support, COUNT-MATCHED to the gate (exp/db >=700)')
    for t in THRS:
        rows = {}
        for arm in ('gate', 'base', 'notears'):
            rp = np.mean([per[c][t][arm]['phys_rate'] for c in per]) * 100
            rc = np.mean([per[c][t][arm]['comb_rate'] for c in per]) * 100
            nm = np.mean([per[c][t][arm]['n'] for c in per])
            rows[arm] = dict(phys_mean=rp, comb_mean=rc, mean_edges=nm)
            log('  |W|>%.1f  %-8s edges=%.1f  physical %.2f%%  (combined %.2f%%)'
                % (t, arm, nm, rp, rc))
        gvn = paired(t, 'gate', 'notears', 'phys_rate')
        gvb = paired(t, 'gate', 'base', 'phys_rate')
        gvn_c = paired(t, 'gate', 'notears', 'comb_rate')
        log('    [phys]  gate-vs-NOTEARS@k %+.2f pp (%d/%d >0, P=%.3g)'
            % (gvn['mean_diff'], gvn['n_win'], gvn['n'], gvn['p']))
        log('    [phys]  gate-vs-base@k    %+.2f pp (%d/%d >0, P=%.3g)'
            % (gvb['mean_diff'], gvb['n_win'], gvb['n'], gvb['p']))
        log('    [comb]  gate-vs-NOTEARS@k %+.2f pp (P=%.3g)  [should reproduce SN4.2]'
            % (gvn_c['mean_diff'], gvn_c['p']))
        summary[str(t)] = dict(arms=rows, gate_vs_notears_phys=gvn,
                               gate_vs_base_phys=gvb, gate_vs_notears_comb=gvn_c)

    json.dump(dict(summary=summary,
                   per={c: {str(t): per[c][t] for t in per[c]} for c in per},
                   n_comb_pairs=len(comb_pairs), n_phys_pairs=len(phys_pairs),
                   channel_cut=CHANNEL_CUT, combined_cut=COMBINED_CUT),
              open(OUT, 'w', encoding='utf-8'), indent=1, default=str)
    log('\nsaved -> %s' % OUT)


if __name__ == '__main__':
    main()
