"""
CAGate Cross-Method Validation: GOLEM + DAGMA + GES
33 TCGA cancers x 6 dimensions x 10 seeds x 2 modes

CRASH-PROOF FEATURES:
- Global try/except in main() — any unhandled crash logs to file
- Heartbeat file — proves process is alive every run
- Flush after every print — real-time progress visible
- GES runs last so fast methods finish first
"""
import sys, os, json, time, glob, traceback
import numpy as np
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(SCRIPT_DIR, '_cross_ckpt.json')
RESULTS = os.path.join(SCRIPT_DIR, '_cross_results.json')
CRASH_LOG = os.path.join(SCRIPT_DIR, '_cross_crash.log')
HEARTBEAT = os.path.join(SCRIPT_DIR, '_cross_heartbeat.txt')

METHODS = ['golem', 'dagma']  # GES runs separately (_cross_ges_bg.py)
SEEDS = [42, 123, 456, 789, 999, 111, 222, 333, 444, 555]
D_SIZES = [30, 50, 100, 150, 200, 300]
L1 = 0.1; ALPHA = 0.5; MAX_K = 8; ITER = 100
DATA_DIR = os.path.join(PKG_ROOT, 'data')

# ═══ Atomic JSON I/O ═══
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
        print(f'[WARN] Corrupt {os.path.basename(path)}, resetting', flush=True)
        return default if default is not None else {}

def heartbeat(msg=''):
    with open(HEARTBEAT, 'w') as f: f.write(f'{time.strftime("%H:%M:%S")} {msg}')

# ═══ Data ═══
def get_cancers():
    cancers = []
    for f in sorted(glob.glob(os.path.join(DATA_DIR, 'TCGA_*_HiSeqV2.tsv'))):
        name = f.split('TCGA_')[1].split('_HiSeqV2')[0]
        cancers.append((name, f))
    return cancers

def load_tcga(filepath, d=300):
    X = []
    with open(filepath, 'r') as f:
        header = f.readline().strip().split('\t'); samples = header[1:]
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

# ═══ Clustering ═══
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

# ═══ GPU ═══
def _dev(d):
    import torch
    if torch.cuda.is_available():
        fm = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        if d <= 200 or fm > 2e9: return torch.device('cuda')
    return torch.device('cpu')

# ═══ GOLEM ═══
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

# ═══ DAGMA ═══
def run_dagma(X, gw, seed):
    import torch; torch.manual_seed(seed); n,d=X.shape; dev=_dev(d)
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    wt = torch.tensor(gw, dtype=torch.float32, device=dev)
    W = torch.zeros(d, d, requires_grad=True, device=dev)
    opt = torch.optim.Adam([W], lr=2e-3)
    for it in range(ITER):
        opt.zero_grad()
        r = Xt - Xt @ W
        mse = (wt.unsqueeze(1) * r.pow(2)).sum() / (2*n)
        I_M = torch.eye(d, device=dev) - W*W
        try: dl = -torch.logdet(I_M)
        except: dl = -torch.logdet(I_M + 1e-6*torch.eye(d, device=dev))
        loss = mse + dl + L1*W.abs().sum()
        loss.backward(); torch.nn.utils.clip_grad_norm_([W], 10.0); opt.step()
        if it%20==0 and it>0 and (W.abs()>1e-6).sum().item()==0 and it>40: break
    e = (W.detach().abs()>1e-6).sum().item()
    if dev.type=='cuda': del Xt,wt,W,opt,r,I_M; torch.cuda.empty_cache()
    return int(e)

# ═══ GES ═══
def run_ges(X, gw=None, seed=42):
    import numpy as np; np.random.seed(seed); d = X.shape[1]
    if gw is not None and len(set(np.round(gw, 2))) > 1:
        labels = cluster_on_residuals(X, np.zeros((d, d)))
        ae = set()
        for c in np.unique(labels):
            m = labels==c; Xc = X[m]
            if len(Xc)<10: continue
            try:
                from causallearn.search.ScoreBased.GES import ges
                rec = ges(Xc); adj = rec['G'].graph
                for i in range(d):
                    for j in range(d):
                        if adj[i,j]==1 and adj[j,i]==-1: ae.add((i,j))
            except: return None
        return len(ae)
    else:
        try:
            from causallearn.search.ScoreBased.GES import ges
            rec = ges(X); adj = rec['G'].graph
            return sum(1 for i in range(d) for j in range(d) if adj[i,j]==1 and adj[j,i]==-1)
        except: return None

# ═══ MAIN ═══
def main():
    heartbeat('start')
    cancers = get_cancers()
    TR = len(cancers)*len(METHODS)*len(D_SIZES)*len(SEEDS)*2
    print(f'Cancer={len(cancers)} Methods={METHODS} d={D_SIZES} Seeds={len(SEEDS)} Total={TR}', flush=True)
    
    ckpt = safe_load(CKPT); results = safe_load(RESULTS)
    d0 = sum(1 for v in ckpt.values() if v.get('done'))
    print(f'Resumed: {d0}/{TR} done', flush=True)
    
    for ci, (cn, fp) in enumerate(cancers):
        heartbeat(f'loading {cn}')
        print(f'\n[{ci+1}/{len(cancers)}] {cn}', end=' ', flush=True)
        try: Xf = load_tcga(fp, d=max(D_SIZES))
        except Exception as e: print(f'SKIP: {e}', flush=True); continue
        
        if Xf.shape[0]<50 or Xf.shape[1]<max(D_SIZES):
            print(f'SKIP n={Xf.shape[0]}', flush=True); continue
        
        print(f'n={Xf.shape[0]}', flush=True); cd = 0
        
        d_list = GES_D if len([m for m in ['ges'] if m in str(METHODS)]) > 0 else D_SIZES
        for dv in D_SIZES:
            X = Xf[:,:dv]
            for idx_mn, mn in enumerate(METHODS):
                seeds_use = GES_SEEDS if mn == 'ges' else SEEDS
            for sd in seeds_use:
                    key = f'{cn}_d{dv}_{mn}_seed{sd}'
                    if key in ckpt: continue
                    heartbeat(f'{cn} d={dv} {mn} s={sd}')
                    try:
                        base = None
                        if mn=='golem': base = run_golem(X, np.ones(len(X)), sd)
                        elif mn=='dagma': base = run_dagma(X, np.ones(len(X)), sd)
                        elif mn=='ges': base = run_ges(X, gw=None, seed=sd)
                        
                        W0 = np.zeros((dv,dv))
                        labs = cluster_on_residuals(X, W0)
                        gw = mad_gate_weights(X, labs)
                        
                        cag = None
                        if mn=='golem': cag = run_golem(X, gw, sd)
                        elif mn=='dagma': cag = run_dagma(X, gw, sd)
                        elif mn=='ges': cag = run_ges(X, gw=gw, seed=sd)
                        
                        base_e = base if base is not None else 0
                        cag_e = cag if cag is not None else 0
                        delta = cag_e - base_e
                        
                        results[key] = {'cancer':cn,'method':mn,'seed':sd,'d':dv,
                                       'base_edges':base,'cagate_edges':cag,'delta':delta,
                                       'n_samples':int(X.shape[0]),'n_clusters':int(len(set(labs))),
                                       'timestamp':time.strftime('%H:%M:%S')}
                        ckpt[key] = {'done':True,'delta':delta}
                        cd += 1
                        
                        tag = '+' if delta>0 else ('=' if delta==0 else '-')
                        print(f'  d={dv} {mn.upper()} s{sd}: {base}->{cag} [{tag}]', flush=True)
                        
                        if cd%5==0:
                            atomic_json(CKPT, ckpt); atomic_json(RESULTS, results)
                    except Exception as e:
                        print(f'  ERR {key}: {str(e)[:100]}', flush=True)
                        ckpt[key] = {'error': str(e)[:200]}
                        atomic_json(CKPT, ckpt)
        
        atomic_json(CKPT, ckpt); atomic_json(RESULTS, results)
    
    print('\n'+'='*60, flush=True)
    for m in METHODS:
        wins = sum(1 for k,v in results.items() if m in k and v.get('delta',0)>0)
        tot = sum(1 for k in results if m in k)
        if tot>0:
            md = np.mean([v['delta'] for k,v in results.items() if m in k])
            print(f'{m.upper()}: {wins}/{tot} wins, mean delta={md:.1f}', flush=True)
    done = sum(1 for v in ckpt.values() if v.get('done'))
    errs = sum(1 for v in ckpt.values() if 'error' in v)
    print(f'Done={done}/{TR} Errors={errs}', flush=True)
    heartbeat('COMPLETE')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        with open(CRASH_LOG, 'w') as f:
            f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")}\n{traceback.format_exc()}')
        print(f'CRASH: {e}', flush=True)
        heartbeat('CRASHED')
        sys.exit(1)
