# -*- coding: utf-8 -*-
"""A-M3 (GO version): split the consensus edge set into its two biological axes
and test whether each axis is over-represented among consensus genes.

Reviewer M3: the consensus hubs mix two objects of different nature -- a shared
cytokine-receptor / composition axis (non-malignant stroma and immune infiltrate)
and a tumour-cell-autonomous mitotic arm.  This script labels every gene by GO
membership with full is_a/part_of closure:

    proliferation  = GO:0007049  cell cycle
    composition    = GO:0004896  cytokine receptor activity

and reports (i) the consensus edges within and between the two axes, and
(ii) whether each axis is over-represented among the consensus genes relative to
the 300-gene panel it was selected from (Fisher exact).

Reads only.  Writes A_M3_axes2.json next to this script.
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
import os, sys, json, gzip
from collections import defaultdict, Counter
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from scipy.stats import fisher_exact

C = DATA_DIR
V = GO_DATA_DIR
OUT = os.path.join(C, 'A_M3_axes2.json')

ROOTS = {'proliferation': 'GO:0007049', 'composition': 'GO:0004896'}


def log(*a):
    print(*a, flush=True)


def parse_obo(path, wanted):
    """Return {root: set(root and all its descendants)} -- the annotation closure."""
    parents = defaultdict(set)
    term_id = None
    with open(path, encoding='utf-8', errors='ignore') as f:
        for ln in f:
            ln = ln.strip()
            if ln == '[Term]':
                term_id = None
            elif ln.startswith('id: GO:'):
                term_id = ln.split()[1]
            elif ln.startswith('is_a: ') and term_id:
                parents[ln.split()[1]].add(term_id)      # parent -> child
            elif ln.startswith('relationship: part_of ') and term_id:
                parents[ln.split()[2]].add(term_id)      # whole -> part
    def desc(t):
        seen, stack = set(), [t]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(parents.get(x, ()))
        return seen
    return {r: desc(r) for r in wanted}


def parse_gaf(path):
    g2go = defaultdict(set)
    with gzip.open(path, 'rt', encoding='utf-8', errors='ignore') as f:
        for ln in f:
            if ln.startswith('!'):
                continue
            p = ln.rstrip('\n').split('\t')
            if len(p) < 5:
                continue
            g2go[p[2].upper()].add(p[4])
    return g2go


def main():
    roots = parse_obo(os.path.join(V, 'go_basic.obo'), list(ROOTS.values()))
    root_terms = {k: roots[v] for k, v in ROOTS.items()}
    log('[GO] proliferation closure %d terms, composition closure %d terms'
        % (len(root_terms['proliferation']), len(root_terms['composition'])))

    g2go = parse_gaf(os.path.join(V, 'goa_human.gaf.gz'))
    log('[GO] %d genes with any annotation' % len(g2go))

    def axis_of(g):
        s = g2go.get(g.upper(), set())
        hit = [k for k, ts in root_terms.items() if s & ts]
        if len(hit) == 2:
            return 'both'
        return hit[0] if hit else 'other'

    panel = [g.upper() for g in json.load(open(os.path.join(C, 'edges_d300.json'), encoding='utf-8'))['genes']]
    cons = json.load(open(os.path.join(C, 'B_hi_edges_d300.json'), encoding='utf-8'))
    cgenes = [g.upper() for g in cons['hi_genes']]
    pairs = [(a.upper(), b.upper()) for a, b in cons['hi_pairs']]
    log('[panel] %d genes | [consensus] %d genes, %d directed pairs'
        % (len(panel), len(cgenes), len(pairs)))

    lab = {g: axis_of(g) for g in set(panel) | set(cgenes)}
    gp = Counter(lab[g] for g in panel)
    gc = Counter(lab[g] for g in cgenes)

    log('\n### axis membership')
    for ax in ('proliferation', 'composition', 'both', 'other'):
        log('  %-14s panel %3d/%d (%.1f%%)   consensus %3d/%d (%.1f%%)'
            % (ax, gp[ax], len(panel), 100 * gp[ax] / len(panel),
               gc[ax], len(cgenes), 100 * gc[ax] / len(cgenes)))

    log('\n### over-representation of each axis among consensus genes (vs panel)')
    over = {}
    for ax in ('proliferation', 'composition'):
        a = gc[ax]; b = len(cgenes) - a
        c_ = gp[ax] - a; d = (len(panel) - gp[ax]) - b
        orr, pv = fisher_exact([[a, b], [c_, d]])
        over[ax] = dict(cons=a, n_cons=len(cgenes), panel=gp[ax], n_panel=len(panel),
                        OR=float(orr), p=float(pv))
        log('  %-14s consensus %d/%d vs panel %d/%d   OR=%.2f  P=%.3g'
            % (ax, a, len(cgenes), gp[ax], len(panel), orr, pv))

    seen = set(); cnt = Counter()
    for a, b in pairs:
        if a == b:
            continue
        k = (a, b) if a < b else (b, a)
        if k in seen:
            continue
        seen.add(k)
        key = tuple(sorted([lab.get(a, 'other'), lab.get(b, 'other')]))
        cnt['%s--%s' % key] += 1
    log('\n### %d undirected consensus edges by axis' % len(seen))
    for k in sorted(cnt, key=lambda x: -cnt[x]):
        log('  %-34s %d' % (k, cnt[k]))

    ax_members = {ax: sorted(g for g in cgenes if lab[g] == ax)
                  for ax in ('proliferation', 'composition')}
    for ax, m in ax_members.items():
        log('  %-14s members (%d): %s' % (ax, len(m), ', '.join(m)))

    json.dump(dict(axis_members=ax_members,
                   panel_counts=dict(gp), consensus_counts=dict(gc),
                   over_rep=over, edges_by_axis=dict(cnt),
                   labels=lab, n_undirected=len(seen)),
              open(OUT, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    log('\nsaved -> %s' % OUT)


if __name__ == '__main__':
    main()
