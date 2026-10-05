# -*- coding: utf-8 -*-
"""Dump the FULL GENIE3 VIM matrices for the 33 real TCGA panels.

Why: the real-data comparison showed GENIE3's top-k edges have ~23 pp higher
STRING support than CAGate's.  Random-forest importances are famously hub-biased
(a few regulators absorb all the importance), and hubs are dense in STRING -- so
the advantage may be a degree artefact.  The degree-matched null needs the actual
selected edge sets, hence the full score matrices.

Checkpoint: one .npz per cancer under b3_genie3/vim/.
"""
import os, sys, json, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Paths are resolved relative to this file so the script runs from a fresh
# clone. External inputs (TCGA panels, STRING v12 raw files) are located via
# the environment variables documented in the repository README.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
C = SCRIPTS_DIR
REPO = REPO_ROOT
WDIR = os.path.join(DATA_DIR, 'mega33_w')
VIMDIR = os.path.join(DATA_DIR, 'genie3', 'vim')
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(SCRIPTS_DIR, 'genie3_vendor'))


def main():
    import numpy as np
    from GENIE3 import GENIE3 as genie3_run
    from _b3_runfit import load as rf_load
    from _b3_genie3 import CANCERS

    os.makedirs(VIMDIR, exist_ok=True)
    print('GENIE3 VIM dump for %d cancers' % len(CANCERS), flush=True)
    t0 = time.time()
    for c in CANCERS:
        fp = os.path.join(VIMDIR, '%s.npz' % c)
        if os.path.exists(fp):
            print('  %-5s: done (skip)' % c, flush=True)
            continue
        js = os.path.join(WDIR, '%s.json' % c)
        if not os.path.exists(js):
            print('  %-5s: no mega33_w json, skip' % c, flush=True)
            continue
        genes = json.load(open(js, encoding='utf-8'))['genes']
        X, _ = rf_load(c, len(genes))
        if X is None:
            print('  %-5s: no raw matrix, skip' % c, flush=True)
            continue
        np.random.seed(0)
        t = time.time()
        G = genie3_run(X, tree_method='RF', K='sqrt', ntrees=1000, nthreads=8)
        tmp = fp + '.tmp.npz'
        np.savez_compressed(tmp, VIM=G, genes=np.array(genes, dtype=object))
        os.replace(tmp, fp)
        print('  %-5s: VIM %s  (%.0fs)' % (c, G.shape, time.time() - t), flush=True)
    print('GENIE3 VIM dump done (%.0fs)' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
