"""Run CAGate prior comparison: Weak vs ENCODE on BRCA d=300. Saves to _prior_ckpt.json."""
import sys, os, json, time
# sys.path removed: run from package root or pip install dependencies
# sys.path removed: notears modules are in package root
import numpy as np
from genomic_causal_trainer_v2 import solve_cdsm_genomic, SolveConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(SCRIPT_DIR, '_prior_ckpt.json')
DATA_DIR = os.path.join(PKG_ROOT, 'data')

# Load or resume
ckpt = {}
if os.path.exists(CKPT):
    with open(CKPT) as f:
        ckpt = json.load(f)
    print(f'Resumed from checkpoint ({len(ckpt)} entries)')

if 'weak_done' not in ckpt or 'encode_done' not in ckpt:
    # Load data
    d = np.load(os.path.join(DATA_DIR, 'tcga_brca_processed.npz'), allow_pickle=True)
    X = d['X'][:, :300]
    gene_names = list(d['gene_names'][:300])
    tf_indices = list(d['tf_indices'])
    weak_prior = d['prior_mask'][:300, :300]
    enc = np.load(os.path.join(DATA_DIR, 'tcga_brca_prior_encode.npz'))
    encode_prior_full = enc['prior']
    encode_prior = encode_prior_full[:300, :300]

    config = SolveConfig(lambda1=0.1, lambda_prior=0.05, max_iter=30, verbose=True)
    results = {}

    # Weak prior
    if 'weak_done' not in ckpt:
        print('\n=== WEAK PRIOR ===')
        t0 = time.time()
        n_prior_edges = int(weak_prior.sum())
        result = solve_cdsm_genomic(X, prior_mask=weak_prior, tf_root_idxs=tf_indices, config=config)
        W = result['W']
        n_total = int((W != 0).sum())
        n_tf_edges = int(sum(1 for i in tf_indices for j in range(300) if W[i, j] != 0))

        # Check specific targets
        target_genes = ['PGR', 'ERBB4', 'SCUBE2']
        target_map = {}
        for tg in target_genes:
            for j in range(300):
                if gene_names[j].upper() == tg:
                    target_map[tg] = j
                    break
        novel_targets = []
        for tg, j in target_map.items():
            found = any(abs(W[i, j]) > 0 for i in tf_indices)
            novel_targets.append((tg, found))

        results['weak'] = {
            'prior_edges': n_prior_edges,
            'cagate_edges': n_total,
            'tf_edges': n_tf_edges,
            'time_s': round(time.time() - t0, 1),
            'novel_targets': {t: f for t, f in novel_targets},
        }
        ckpt['weak_done'] = True
        ckpt['weak_result'] = results['weak']
        ckpt['gene_names'] = gene_names[:50]
        ckpt['tf_names'] = [gene_names[i] for i in tf_indices[:50]]
        with open(CKPT, 'w') as f:
            json.dump(ckpt, f, indent=2)
        print(f'  Weak: prior={n_prior_edges}, cagate={n_total}, time={time.time()-t0:.1f}s')

    # ENCODE prior
    if 'encode_done' not in ckpt:
        print('\n=== ENCODE PRIOR ===')
        t0 = time.time()
        n_prior_edges = int(encode_prior.sum())
        result = solve_cdsm_genomic(X, prior_mask=encode_prior, tf_root_idxs=tf_indices, config=config)
        W = result['W']
        n_total = int((W != 0).sum())
        n_tf_edges = int(sum(1 for i in tf_indices for j in range(300) if W[i, j] != 0))

        # Check novel targets
        target_genes = ['PGR', 'ERBB4', 'SCUBE2']
        target_map = {}
        for tg in target_genes:
            for j in range(300):
                if gene_names[j].upper() == tg:
                    target_map[tg] = j
                    break
        novel_targets = []
        for tg, j in target_map.items():
            found = any(abs(W[i, j]) > 0 for i in tf_indices)
            novel_targets.append((tg, found))

        results['encode'] = {
            'prior_edges': n_prior_edges,
            'cagate_edges': n_total,
            'tf_edges': n_tf_edges,
            'time_s': round(time.time() - t0, 1),
            'novel_targets': {t: f for t, f in novel_targets},
        }
        ckpt['encode_done'] = True
        ckpt['encode_result'] = results['encode']
        with open(CKPT, 'w') as f:
            json.dump(ckpt, f, indent=2)
        print(f'  ENCODE: prior={n_prior_edges}, cagate={n_total}, time={time.time()-t0:.1f}s')

print('\n[DONE] Prior comparison complete')
print(json.dumps({k: ckpt[k] for k in ['weak_result', 'encode_result'] if k in ckpt}, indent=2))
