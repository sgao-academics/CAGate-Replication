"""
CAGate Cross-Method: GOLEM only, all cancers, d=100/150/200/300.
Resumes from _cross_ckpt.json — skips already-done keys.
"""
import sys, os, json, time, glob, traceback
import numpy as np
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(SCRIPT_DIR, '_cross_ckpt.json')
RESULTS = os.path.join(SCRIPT_DIR, '_cross_results.json')
HEARTBEAT = os.path.join(SCRIPT_DIR, '_cross_heartbeat.txt')

METHOD = 'golem'
SEEDS = [42, 123, 456, 789, 999, 111, 222, 333, 444, 555]
D_SIZES = [30, 50, 100, 150, 200, 300]  # full sweep, skips already-done via ckpt
L1 = 0.1; ALPHA = 0.5; MAX_K = 8; ITER = 100
DATA_DIR = os.path.join(PKG_ROOT, 'data')

def atomic_json(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f: json.dump(data, f, indent=2)
    try: os.replace(tmp, path)
    except: os.rename(tmp, path)

def safe_load(path, default=None):
    if not os.path.exists(path): return default if default is not None else {}
    try:
        with open(path) as f: return json.load(f)
    except:
        return default if default is not None else {}

def heartbeat(msg=''):
    with open(HEARTBEAT, 'w') as f: f.write(f'{time.strftime("%H:%M:%S")} {msg}')

def get_cancers():
    cancers = []
    for f in sorted(glob.glob(os.path.join(DATA_DIR, 'TCGA_*_HiSeqV2.tsv'))):
        name = f.split('TCGA_')[1].split('_HiSeqV2')[0]
        cancers.append((name, f))
    return cancers

def load_tcga(filepath, d=300):
    X = []
    with open(filepath, 'r') as f:
        header = f.readline().strip().split('\t')
        samples = header[1:]
        for line in f:
            parts = line.strip().split('\t')
            if parts[0] and not parts[0].startswith('?'):
                try:
                    vals = [float(x) for x in parts[1:]]
                    if len(vals) == len(samples): X.append(vals)
                except: continue
    X = np.array(X).T
    if X.shape[1] > d:
        variances = np.var(X, axis=0)
        X = X[:, np.argsort(variances)[-d:]]
    return X

def cluster_on_residuals(X, W, max_k=8):
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    R = X - X @ W
    try:
        nc = min(X.shape[1], min(10, X.shape[0]//2))
        R_pca = PCA(n_components=nc).fit_transform(R)
        mk = min(max_k, len(X)//5)
        if mk < 2: return np.zeros(len(X), dtype=int)
        bk, bs = 2, -1
        for k in range(2, mk+1):
            km = KMeans(n_clusters=k, n_init=3, random_state=42)
            labs = km.fit_predict(R_pca)
            if len(set(labs)) < 2: continue
            sc = silhouette_score(R_pca, labs)
            if sc > bs: bk, bs = k, sc
        return KMeans(n_clusters=bk, n_init=10, random_state=42).fit_predict(R_pca)
    except: return np.zeros(len(X), dtype=int)

def mad_gate_weights(X, labels):
    w = np.ones(len(X))
    for c in np.unique(labels):
        m = labels == c; Xc = X[m]
        if len(Xc) > 1:
            mad = np.median(np.abs(Xc - np.median(Xc, axis=0)), axis=0)
            g = 1 - np.exp(-ALPHA * np.mean(mad))
        else: g = 0.5
        w[m] = max(0.01, min(g, 0.99))
    return w

def _dev(d):
    import torch
    if torch.cuda.is_available():
        fm = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        if d <= 200 or fm > 2e9: return torch.device('cuda')
    return torch.device('cpu')

def run_golem(X, gw, seed):
    import torch; n, d = X.shape; dev = _dev(d)
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    wt = torch.tensor(gw, dtype=torch.float32, device=dev)
    W = torch.zeros(d, d, requires_grad=True, device=dev)
    opt = torch.optim.Adam([W], lr=2e-3)
    for it in range(ITER):
        opt.zero_grad()
        r = Xt - Xt @ W
        mse = (wt.unsqueeze(1) * r.pow(2)).sum() / (2*n)
        M = (torch.eye(d, device=dev) - W).t() @ (torch.eye(d, device=dev) - W)
        try: ld = 0.5*torch.logdet(M)
        except: ld = 0.5*torch.logdet(M + 1e-6*torch.eye(d, device=dev))
        loss = mse + ld + L1*W.abs().sum()
        loss.backward(); torch.nn.utils.clip_grad_norm_([W], 10.0); opt.step()
        if it%20==0 and it>0 and (W.abs()>1e-6).sum().item()==0 and it>40: break
    e = (W.detach().abs()>1e-6).sum().item()
    if dev.type=='cuda': del Xt,wt,W,opt,r,M; torch.cuda.empty_cache()
    return int(e)

def main():
    heartbeat('start-golem')
    cancers = get_cancers()
    total = len(cancers) * len(D_SIZES) * len(SEEDS)
    print(f'GOLEM: cancers={len(cancers)} d={D_SIZES} seeds={len(SEEDS)} total={total}', flush=True)

    ckpt = safe_load(CKPT); results = safe_load(RESULTS)

    for ci, (cn, fp) in enumerate(cancers):
        heartbeat(f'loading {cn}')
        print(f'\n[{ci+1}/{len(cancers)}] {cn}', end=' ', flush=True)
        try: Xf = load_tcga(fp, d=max(D_SIZES))
        except Exception as e: print(f'SKIP: {e}', flush=True); continue

        if Xf.shape[0] < 50 or Xf.shape[1] < max(D_SIZES):
            print(f'SKIP n={Xf.shape[0]} d={Xf.shape[1]}', flush=True); continue

        print(f'n={Xf.shape[0]}', flush=True)

        for dv in D_SIZES:
            X = Xf[:,:dv]
            for sd in SEEDS:
                key = f'{cn}_d{dv}_{METHOD}_seed{sd}'
                if key in ckpt:
                    continue  # Skip already done
                heartbeat(f'{cn} d={dv} {METHOD} s={sd}')
                try:
                    # Base GOLEM (uniform weights)
                    base = run_golem(X, np.ones(len(X)), sd)

                    # CAGate-enhanced GOLEM
                    W0 = np.zeros((dv, dv))
                    labs = cluster_on_residuals(X, W0)
                    gw = mad_gate_weights(X, labs)
                    cag = run_golem(X, gw, sd)

                    base_e = base if base is not None else 0
                    cag_e = cag if cag is not None else 0
                    delta = cag_e - base_e

                    results[key] = {
                        'cancer': cn, 'method': METHOD, 'seed': sd, 'd': dv,
                        'base_edges': base_e, 'cagate_edges': cag_e, 'delta': delta,
                        'n_samples': int(X.shape[0]), 'n_clusters': int(len(set(labs))),
                        'timestamp': time.strftime('%H:%M:%S')
                    }
                    ckpt[key] = {'done': True, 'delta': delta}

                    tag = '+' if delta > 0 else ('=' if delta == 0 else '-')
                    print(f'  d={dv} s{sd}: base={base_e} cag={cag_e} delta={delta} [{tag}]', flush=True)

                    atomic_json(CKPT, ckpt)
                    atomic_json(RESULTS, results)
                except Exception as e:
                    print(f'  ERR {key}: {str(e)[:100]}', flush=True)
                    ckpt[key] = {'error': str(e)[:200]}
                    atomic_json(CKPT, ckpt)

        atomic_json(CKPT, ckpt)
        atomic_json(RESULTS, results)

    # Summary
    done = sum(1 for v in ckpt.values() if v.get('done'))
    wins = sum(1 for k, v in results.items() if METHOD in k and v.get('delta', 0) > 0)
    print(f'\nGOLEM DONE: {done} runs, {wins} wins', flush=True)
    heartbeat('COMPLETE')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        with open(os.path.join(SCRIPT_DIR, '_cross_golem_crash.log'), 'w') as f:
            f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")}\n{traceback.format_exc()}')
        print(f'CRASH: {e}', flush=True)
        heartbeat('CRASHED')
        sys.exit(1)
