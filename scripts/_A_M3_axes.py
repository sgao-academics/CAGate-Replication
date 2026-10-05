# -*- coding: utf-8 -*-
"""A-M3: split the consensus edge set into its two biological axes.

Reviewer M3 points out that the consensus hubs mix two objects of different
nature: a shared cytokine-receptor / composition axis (non-malignant stroma and
immune infiltrate) and a tumour-cell-autonomous mitotic arm.  This script
classifies every consensus gene by KEGG module membership -- hsa04060
(Cytokine-cytokine receptor interaction) and hsa04110 (Cell cycle), the same
two maps used for the Supplementary Figure S4 colouring -- and recounts the
consensus edges within and between those modules.

Reads only: B_hi_edges_{d300,np300}.json (the consensus edge lists) and the
KEGG flat files.  Writes A_M3_axes.json next to this script.
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
import os, sys, json
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

C = DATA_DIR
V = GO_DATA_DIR
OUT = os.path.join(C, 'A_M3_axes.json')

WANT_PW = {'Cytokine-cytokine receptor interaction': 'cytokine-receptor',
           'Cell cycle': 'cell-cycle'}


def J(f):
    return json.load(open(os.path.join(C, f), encoding='utf-8'))


def build_g2mod():
    ent2sym = {}
    with open(os.path.join(V, 'kegg_hsa_genes.txt'), encoding='utf-8', errors='ignore') as f:
        for ln in f:
            p = ln.rstrip('\n').split('\t')
            if len(p) >= 4:
                s = p[3].split(';')[0].split(',')[0].strip().upper()
                if s:
                    ent2sym[p[0].strip()] = s
    pw2ent = defaultdict(set)
    with open(os.path.join(V, 'kegg_gene_pathway.txt'), encoding='utf-8', errors='ignore') as f:
        for ln in f:
            p = ln.rstrip('\n').split('\t')
            if len(p) >= 2:
                pw2ent[p[0].strip()].add(p[1].strip())
    pwname = {}
    with open(os.path.join(V, 'kegg_hsa_pathways.txt'), encoding='utf-8', errors='ignore') as f:
        for ln in f:
            p = ln.rstrip('\n').split('\t')
            if len(p) >= 2:
                pwname[p[0].strip().split(':')[-1]] = p[1].strip().split(' - ')[0]
    g2mod = defaultdict(set)
    for pw, ents in pw2ent.items():
        lab = WANT_PW.get(pwname.get(pw.strip().split(':')[-1], ''))
        if not lab:
            continue
        for e in ents:
            if e in ent2sym:
                g2mod[ent2sym[e]].add(lab)
    return g2mod


def lab_of(g2mod, g):
    m = g2mod.get(g.upper())
    if not m:
        return 'other'
    if len(m) == 2:
        return 'both'
    return sorted(m)[0]


def main():
    g2mod = build_g2mod()
    out = {}
    for tag in ('d300', 'np300'):
        j = J('B_hi_edges_%s.json' % tag)
        genes = [g.upper() for g in j['hi_genes']]
        pairs = [(a.upper(), b.upper()) for a, b in j['hi_pairs']]
        lab = {g: lab_of(g2mod, g) for g in genes}
        # undirected count of distinct pairs
        seen = set()
        cnt = Counter()
        for a, b in pairs:
            if a == b:
                continue
            k = (a, b) if a < b else (b, a)
            if k in seen:
                continue
            seen.add(k)
            la, lb = lab.get(a, 'other'), lab.get(b, 'other')
            key = tuple(sorted([la, lb]))
            cnt['%s--%s' % key] += 1
        gene_cnt = Counter(lab.values())
        out[tag] = dict(d=j['d'], n_genes=len(genes), n_edges=len(seen),
                        gene_by_axis=dict(gene_cnt),
                        edges_by_axis=dict(cnt),
                        labels=lab)
        print('=== %s  d=%d  genes=%d  undirected consensus edges=%d'
              % (tag, j['d'], len(genes), len(seen)))
        print('    genes by axis :', dict(gene_cnt))
        for k in sorted(cnt, key=lambda x: -cnt[x]):
            print('    edges %-26s %d' % (k, cnt[k]))
        # list the members of each axis
        for ax in ('cytokine-receptor', 'cell-cycle'):
            mem = sorted(g for g in genes if lab[g] == ax)
            print('    %-18s (%d): %s' % (ax, len(mem), ', '.join(mem)))
        other = sorted(g for g in genes if lab[g] == 'other')
        print('    other              (%d): %s' % (len(other), ', '.join(other)))

    json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('\nsaved -> %s' % OUT)


if __name__ == '__main__':
    main()
