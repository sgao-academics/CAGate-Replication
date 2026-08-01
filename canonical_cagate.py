"""
Ablation V3 Fortified: Cluster-Aware Gradient with crash-proof execution.
- Singular matrix → retry with perturbed Wt
- NOTEARS divergence → catch, log, skip seed
- Per-seed checkpoint (never lose work)
- Error log for every failed seed
- Zero silent failures. Every crash leaves a trace.

Usage: python ablation_v3_fortified.py
"""
import sys, json, time, os, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from cluster_aware_loss import cluster_gate

OUT = os.path.dirname(os.path.abspath(__file__)) + '/benchmark_results'
os.makedirs(OUT, exist_ok=True)
ERROR_LOG = os.path.join(OUT, 'v3_fortified_errors.jsonl')
TIMING_LOG = os.path.join(OUT, 'v3_fortified_timing.jsonl')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'DEVICE: {device}', flush=True)
if device.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)', flush=True)

# ══════════════════════════════════════════════════════════════════
#  FORTIFIED DATA GENERATOR — handles singular matrices
# ══════════════════════════════════════════════════════════════════

def gen_cluster(d=10, nc=5, sp=100, max_retries=10):
    """Generate clustered DAG data with singular-matrix resilience."""
    for attempt in range(max_retries):
        Wt = torch.zeros(d, d)
        e = 0
        while e < 2 * d:
            i, j = np.random.randint(0, d, 2)
            if i < j and Wt[i, j] == 0:
                Wt[i, j] = np.random.uniform(0.5, 2.0) * np.random.choice([-1, 1])
                e += 1

        # Ensure DAG structure: zero upper-triangular → always invertible
        # Permute columns to random topological order
        perm = torch.randperm(d)
        Wt_dag = Wt[perm][:, perm]

        try:
            M = torch.eye(d) - Wt_dag.T
            inv_M = torch.linalg.inv(M)
        except RuntimeError:
            # Singular → add small diagonal perturbation and retry
            if attempt < max_retries - 1:
                continue
            # Last resort: use pseudoinverse
            inv_M = torch.linalg.pinv(M)

        n = nc * sp
        X = torch.zeros(n, d)
        cid = torch.zeros(n, dtype=torch.long)
        for c in range(nc):
            cn = 1.0 * (0.3 + 2.0 * c / (nc - 1)) if nc > 1 else 0.3
            idx = slice(c * sp, (c + 1) * sp)
            Xc = torch.randn(sp, d) @ inv_M
            Xc += cn * torch.randn(sp, d)
            X[idx] = Xc
            cid[idx] = c
        return X, Wt_dag, cid

    # Should never reach here (max_retries loop always returns)
    raise RuntimeError(f"gen_cluster failed after {max_retries} retries")


# ══════════════════════════════════════════════════════════════════
#  FORTIFIED NOTEARS — catches divergence
# ══════════════════════════════════════════════════════════════════

def run_notears(X, gate=False, cid=None, max_rho=1e12,
                lr=0.001, n_outer=40, n_inner=400, l1_penalty=0.01, gate_alpha=0.5):
    """Run NOTEARS with NaN/Inf guard.
    Hyperparameters are config-dependent (passed by caller).
    """
    d = X.shape[1]
    Xg = X.to(device)
    W = torch.zeros(d, d, requires_grad=True, device=device)
    rho, alpha = 1.0, 0.0
    opt = torch.optim.Adam([W], lr=lr)

    for o in range(n_outer):
        for i in range(n_inner):
            opt.zero_grad()
            M = torch.eye(d, device=device) - W
            sq = (Xg @ M.T) ** 2

            if gate:
                res = sq.mean(dim=1)
                gates, _ = cluster_gate(res, cid.to(device), alpha=gate_alpha)
                loss_d = (gates * sq.sum(dim=1)).sum() / gates.sum().clamp(min=1)
            else:
                loss_d = sq.mean()

            h = torch.trace(torch.linalg.matrix_exp(W * W)) - d
            loss = loss_d + 0.5 * rho * h ** 2 + alpha * h + l1_penalty * torch.sum(torch.abs(W))

            # NaN/Inf guard
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"NOTEARS loss diverged (NaN/Inf) at outer={o} inner={i}")

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([W], max_norm=10.0)

            opt.step()

        with torch.no_grad():
            hv = (torch.trace(torch.linalg.matrix_exp(W * W)) - d).item()

        if torch.isnan(torch.tensor(hv)) or torch.isinf(torch.tensor(hv)):
            raise RuntimeError(f"NOTEARS h diverged (NaN/Inf) at outer={o}")

        alpha = alpha + rho * hv if abs(hv) > 1e-6 else alpha
        rho = min(5 * rho, max_rho)

    return W.detach().cpu()


# ══════════════════════════════════════════════════════════════════
#  F1 SCORE
# ══════════════════════════════════════════════════════════════════

def f1(W_e, W_t):
    Wb = (torch.abs(W_e) > 0.3).float()
    Tb = (torch.abs(W_t) > 0).float()
    tp = ((Wb == 1) & (Tb == 1)).sum().item()
    fp = ((Wb == 1) & (Tb == 0)).sum().item()
    fn = ((Wb == 0) & (Tb == 1)).sum().item()
    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0
    return 2 * p * r / (p + r) if p + r > 0 else 0


# ══════════════════════════════════════════════════════════════════
#  ERROR LOGGING
# ══════════════════════════════════════════════════════════════════

def log_error(seed, cfg_label, stage, exc, tb_text):
    entry = {
        'seed': seed, 'config': cfg_label, 'stage': stage,
        'error': str(exc), 'traceback': tb_text,
        'time': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(ERROR_LOG, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f'  [ERROR] seed={seed} stage={stage}: {exc}', flush=True)


# ══════════════════════════════════════════════════════════════════
#  MAIN: 30 seeds × 3 configs, fully fortified
# ══════════════════════════════════════════════════════════════════

CONFIGS = [
    {'label': 'd15_c5',  'd': 15, 'nc': 5,  'sp': 100},
    {'label': 'd15_c8',  'd': 15, 'nc': 8,  'sp': 60},
    {'label': 'd20_c6',  'd': 20, 'nc': 6,  'sp': 80},
]
N_SEEDS = 20

all_results = []

for cfg in CONFIGS:
    label = cfg['label']
    d, nc, sp = cfg['d'], cfg['nc'], cfg['sp']
    ckpt_path = os.path.join(OUT, f'v3_ckpt_{label}.json')
    config_path = os.path.join(OUT, f'v3_config_{label}.json')

    print(f'\n{"="*60}', flush=True)
    print(f'CONFIG: {label}  d={d} nc={nc} n={nc*sp}', flush=True)
    print(f'{"="*60}', flush=True)

    f1_base, f1_gate = [], []
    seeds_done, seeds_failed = 0, 0
    start = 0

    # Resume from checkpoint
    if os.path.exists(ckpt_path):
        try:
            ckpt = json.load(open(ckpt_path))
            f1_base = ckpt.get('f1_base', [])
            f1_gate = ckpt.get('f1_gate', [])
            start = len(f1_base)
            if start > 0:
                print(f'  Resuming from seed {start}/{N_SEEDS} ({len(f1_base)} completed)', flush=True)
        except:
            print(f'  [WARN] Corrupted checkpoint, starting fresh', flush=True)

    # ════ Config-dependent hyperparameters ════
    if d == 20:
        # d=20: harder optimization → lower LR, more iterations, softer gate
        hp_lr = 0.0003
        hp_outer = 60
        hp_inner = 500
        hp_l1 = 0.005
        hp_gate_alpha = 0.3
        print(f'  [HP] d=20 → lr={hp_lr} outer={hp_outer} inner={hp_inner} '
              f'l1={hp_l1} gate_alpha={hp_gate_alpha}', flush=True)
    else:
        # d=15: default (proven)
        hp_lr = 0.001
        hp_outer = 40
        hp_inner = 400
        hp_l1 = 0.01
        hp_gate_alpha = 0.5
    # ════════════════════════════════════════

    for seed in range(start, N_SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        t0 = time.time()

        try:
            # Stage 1: Generate data
            X, Wt, cid = gen_cluster(d, nc, sp)
        except Exception as e:
            log_error(seed, label, 'gen_cluster', e, traceback.format_exc())
            seeds_failed += 1
            continue

        # Stage 2: Baseline NOTEARS
        try:
            Wb = run_notears(X, gate=False,
                                 lr=hp_lr, n_outer=hp_outer, n_inner=hp_inner,
                                 l1_penalty=hp_l1)
            fb = f1(Wb, Wt)
        except Exception as e:
            log_error(seed, label, 'run_notears(baseline)', e, traceback.format_exc())
            seeds_failed += 1
            continue

        # Stage 3: Cluster-Aware NOTEARS
        try:
            Wg = run_notears(X, gate=True, cid=cid,
                                  lr=hp_lr, n_outer=hp_outer, n_inner=hp_inner,
                                  l1_penalty=hp_l1, gate_alpha=hp_gate_alpha)
            fg = f1(Wg, Wt)
        except Exception as e:
            log_error(seed, label, 'run_notears(cluster_aware)', e, traceback.format_exc())
            # Baseline succeeded even if gate failed — save baseline
            f1_base.append(fb)
            seeds_done += 1
            seeds_failed += 1
            dt = time.time() - t0
            mb = np.mean(f1_base)
            print(f'  seed {seed}/{N_SEEDS-1} [{dt:.0f}s] base={fb:.4f} gate=FAILED '
                  f'| mean base={mb:.4f}', flush=True)
            with open(ckpt_path, 'w') as f:
                json.dump({'f1_base': f1_base, 'f1_gate': f1_gate,
                           'seeds_done': seeds_done, 'seeds_failed': seeds_failed}, f)
            continue

        # Both succeeded
        f1_base.append(fb)
        f1_gate.append(fg)
        seeds_done += 1

        dt = time.time() - t0
        mb = np.mean(f1_base)
        mg = np.mean(f1_gate) if f1_gate else 0
        delta_str = f'{mg - mb:+.4f}' if f1_gate else 'N/A'
        print(f'  seed {seed}/{N_SEEDS-1} [{dt:.0f}s] base={fb:.4f} gate={fg:.4f} '
              f'| mean base={mb:.4f} gate={mg:.4f} Δ={delta_str}', flush=True)

        # Save checkpoint after EVERY seed
        with open(ckpt_path, 'w') as f:
            json.dump({'f1_base': f1_base, 'f1_gate': f1_gate,
                       'seeds_done': seeds_done, 'seeds_failed': seeds_failed}, f)

        # Timing log for diagnostics
        with open(TIMING_LOG, 'a') as f:
            f.write(json.dumps({'config': label, 'seed': seed, 'time_s': round(dt, 1),
                                'f1_base': round(fb, 4), 'f1_gate': round(fg, 4)}) + '\n')

    # ── Config complete ──
    n_ok = seeds_done - seeds_failed
    r = {
        'config': cfg,
        'n_seeds_target': N_SEEDS,
        'n_seeds_completed': seeds_done,
        'n_seeds_failed': seeds_failed,
        'n_seeds_valid': n_ok,
        'baseline_mean': round(np.mean(f1_base), 4) if f1_base else None,
        'cluster_aware_mean': round(np.mean(f1_gate), 4) if f1_gate else None,
        'delta_mean': round(np.mean(f1_gate) - np.mean(f1_base), 4) if f1_gate and f1_base else None,
        'f1_base_all': [round(x, 4) for x in f1_base],
        'f1_gate_all': [round(x, 4) for x in f1_gate],
    }
    all_results.append(r)

    print(f'\n  FINAL ({label}): {r["baseline_mean"]} → {r["cluster_aware_mean"]} '
          f'(Δ={r["delta_mean"]}) [{n_ok}/{N_SEEDS} valid seeds]', flush=True)

    # Save config-level result
    with open(config_path, 'w') as f:
        json.dump(r, f, indent=2)

# ══════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════

summary_path = os.path.join(OUT, 'ablation_v3_fortified.json')
with open(summary_path, 'w') as f:
    json.dump({'results': all_results, 'generated': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)

print(f'\n{"="*60}', flush=True)
print(f'ALL CONFIGS DONE', flush=True)
print(f'{"="*60}', flush=True)
for r in all_results:
    c = r['config']
    print(f'  {c["label"]}: {r["baseline_mean"]} → {r["cluster_aware_mean"]} '
          f'(Δ={r["delta_mean"]}) [{r["n_seeds_valid"]}/{r["n_seeds_target"]} valid]', flush=True)

# Check error log
if os.path.exists(ERROR_LOG):
    n_errors = sum(1 for _ in open(ERROR_LOG, encoding='utf-8'))
    if n_errors > 0:
        print(f'\n  ⚠ {n_errors} errors logged to {ERROR_LOG}', flush=True)
    else:
        os.remove(ERROR_LOG)

print(f'\nSummary:   {summary_path}', flush=True)
for r in all_results:
    cfg_label = r['config']['label']
    print(f'Config:    {os.path.join(OUT, "v3_config_" + cfg_label + ".json")}', flush=True)
print(f'Checkpoints: {os.path.join(OUT, "v3_ckpt_*.json")}', flush=True)
if os.path.exists(ERROR_LOG):
    print(f'Errors:    {ERROR_LOG}', flush=True)
print('\n=== FORTIFIED RUN COMPLETE ===', flush=True)
