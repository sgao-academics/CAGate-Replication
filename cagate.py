"""
═══════════════════════════════════════════════════════════════════════
CAGate solver: cluster-aware gated differentiable causal discovery.

The estimator combines three ingredients:

  * the NOTEARS residual convention, R = X - X W, so that W is oriented as
    in X = X W_true + Z; no transposition of the adjacency is involved.
  * a doubled-variable L1 scheme solved by L-BFGS-B with an augmented
    Lagrangian (rho <- 10 rho, rho_max = 1e16, h_tol = 1e-8), run to
    convergence rather than for a fixed number of steps.
  * the cluster-aware gate, applied IRLS-style: the gate weights are
    recomputed from the current residuals once per outer iteration and then
    held fixed while the inner weighted least-squares subproblem is solved
    exactly.

Edge counts are invariant to transposition, whereas orientation-sensitive
quantities (TPR / SHD / in- and out-degree / parent->child) depend on the
residual convention above.
═══════════════════════════════════════════════════════════════════════
"""
import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt


# ─────────────────────────────────────────────────────────────────────
#  Cluster gate (exact NumPy re-implementation of cluster_aware_loss.cluster_gate)
# ─────────────────────────────────────────────────────────────────────

def cluster_gate_np(res, cid, alpha=0.5):
    """res: (n,) per-sample residual energy; cid: (n,) int cluster labels.
    gate_g = 1 / (1 + exp(-alpha * (sigma_bar - sigma_g))).
    """
    cid = np.asarray(cid).astype(int)
    nc = int(cid.max()) + 1
    sigma = np.zeros(nc)
    for g in range(nc):
        m = cid == g
        if m.sum() > 1:
            sigma[g] = res[m].std(ddof=1)
    pos = sigma[sigma > 0]
    sigma_bar = float(np.median(pos)) if pos.size else 1.0
    gpc = 1.0 / (1.0 + np.exp(-alpha * (sigma_bar - sigma)))
    return gpc[cid], {'sigma': sigma, 'sigma_bar': sigma_bar, 'gates_per_cluster': gpc}


# ─────────────────────────────────────────────────────────────────────
#  DAG constraint  h(W) = Tr(e^{W⊙W}) - d
# ─────────────────────────────────────────────────────────────────────

def h_dag(W):
    M = W * W
    E = slin.expm(M)
    d = W.shape[0]
    return np.trace(E) - d, E.T * W * 2


# ─────────────────────────────────────────────────────────────────────
#  Corrected CAGate:  L-BFGS-B + correct orientation + cluster gate
# ─────────────────────────────────────────────────────────────────────

def solve_cagate_lbfgsb(X, cid=None, l1=0.1, gate_alpha=0.5,
                        max_outer=100, h_tol=1e-8, rho_max=1e16,
                        w_threshold=0.3, use_gate=True,
                        max_inner=20, maxiter=1000, verbose=False,
                        prior_mask=None, lambda_prior=0.0,
                        refit_gates=True, n_clusters=None, refit_clusters=False,
                        cluster_seed=0, pca_var=0.8):
    """Solve  min_W  1/(2n) Σ_i g_i ||x_i - x_i W||^2 + λ1 ||W||_1
              s.t. h(W) = 0, diag(W) = 0.

    g_i = cluster gate (per sample), refreshed each outer iteration from the
    current residuals (IRLS).  Orientation is CORRECT: R = X - X W.

    refit_clusters=False (default, used for the reported runs): cluster labels
      are supplied once by the caller.
    refit_clusters=True: labels are re-estimated from the CURRENT residuals at
      every outer iteration (PCA to `pca_var` explained variance, then K-means
      with `n_clusters`), i.e. clustering and structure learning co-evolve.
    """
    if refit_clusters:
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans as _KM
        n_clusters = int(n_clusters or (int(np.max(cid)) + 1 if cid is not None else 5))
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)

    bound = np.ones((d, d), dtype=np.float64)
    np.fill_diagonal(bound, 0.0)

    def _adj(w):
        return (w[:d * d] - w[d * d:]).reshape(d, d)

    gates = np.ones(n)

    # bounds: two copies (positive / negative), zeros where masked
    bnds = []
    for _ in range(2):
        for i in range(d):
            for j in range(d):
                bnds.append((0, 0) if bound[i, j] == 0 else (0, None))

    w_est = np.zeros(2 * d * d)
    rho, alpha, h = 1.0, 0.0, np.inf

    for it in range(max_outer):
        if use_gate and refit_gates:
            Wc = _adj(w_est) * bound
            R = Xc - Xc @ Wc
            res = (R ** 2).mean(axis=1)
            if refit_clusters:
                # re-partition samples on the CURRENT residuals before gating
                Rp = R - R.mean(axis=0, keepdims=True)
                try:
                    Z = PCA(n_components=pca_var, svd_solver='full').fit_transform(Rp)
                except Exception:
                    Z = Rp
                Z = np.nan_to_num(Z)
                k = int(min(n_clusters, max(2, n - 1)))
                cid = _KM(n_clusters=k, random_state=cluster_seed,
                          n_init=10).fit_predict(Z)
            if cid is not None:
                gates, _ = cluster_gate_np(res, np.asarray(cid), gate_alpha)

        def _func(w):
            W = _adj(w) * bound
            R = Xc - Xc @ W
            wR = gates[:, None] * R
            mse = 0.5 / n * (wR * R).sum()
            G_mse = -1.0 / n * Xc.T @ wR
            hh, G_h = h_dag(W)
            obj = mse + 0.5 * rho * hh * hh + alpha * hh + l1 * w.sum()
            Gs = G_mse + (rho * hh + alpha) * G_h
            if lambda_prior and prior_mask is not None:
                obj += lambda_prior * np.abs(W * (1 - prior_mask)).sum()
                Gs = Gs + lambda_prior * np.sign(W) * (1 - prior_mask)
            Gs = Gs * bound
            g_obj = np.concatenate((Gs.flatten() + l1, -Gs.flatten() + l1))
            return obj, g_obj

        w_new, h_new = None, None
        inner = 0
        while rho < rho_max and inner < max_inner:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True,
                                bounds=bnds,
                                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
            w_new = sol.x
            h_new, _ = h_dag(_adj(w_new) * bound)
            inner += 1
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if verbose:
            nz = int((np.abs(_adj(w_est) * bound) > w_threshold).sum())
            print('  it%3d h=%.2e rho=%.1e nz>%.1f=%d' % (it, h, rho, w_threshold, nz))
        if h <= h_tol or rho >= rho_max:
            break

    W_est = _adj(w_est) * bound
    counts = {('%g' % t): int((np.abs(W_est) > t).sum())
              for t in (1e-6, 1e-3, 1e-2, 0.05, 0.1, 0.3)}
    W_thr = W_est.copy()
    W_thr[np.abs(W_thr) < w_threshold] = 0.0
    return {'W': W_est, 'W_thr': W_thr,
            'n_edges': int((np.abs(W_est) > w_threshold).sum()),
            'h': h, 'counts': counts, 'gates': gates}


# ─────────────────────────────────────────────────────────────────────
#  Data generators
# ─────────────────────────────────────────────────────────────────────

def gen_cluster(d=10, nc=5, sp=100, seed=0, transpose=False):
    """Clustered linear-SEM data.

    transpose=False  ->  X = X Wt + Z   (NOTEARS convention; W matches Wt)
    transpose=True   ->  X = X Wt^T + Z (transposed convention)
    """
    rng = np.random.RandomState(seed)
    Wt = np.zeros((d, d))
    e = 0
    while e < 2 * d:
        i, j = rng.randint(0, d, 2)
        if i < j and Wt[i, j] == 0:
            Wt[i, j] = rng.uniform(0.5, 2.0) * rng.choice([-1, 1])
            e += 1
    perm = rng.permutation(d)
    Wt_dag = Wt[perm][:, perm]
    M = np.eye(d) - (Wt_dag.T if transpose else Wt_dag)
    inv_M = np.linalg.inv(M)
    n = nc * sp
    X = np.zeros((n, d))
    cid = np.zeros(n, dtype=int)
    for c in range(nc):
        cn = 1.0 * (0.3 + 2.0 * c / (nc - 1)) if nc > 1 else 0.3
        idx = slice(c * sp, (c + 1) * sp)
        Xc = rng.randn(sp, d) @ inv_M
        Xc += cn * rng.randn(sp, d)
        X[idx] = Xc
        cid[idx] = c
    return X, Wt_dag, cid


def f1(W_e, W_t, thr=0.3):
    Wb = (np.abs(W_e) > thr)
    Tb = (np.abs(W_t) > 0)
    tp = int((Wb & Tb).sum()); fp = int((Wb & ~Tb).sum()); fn = int((~Wb & Tb).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0, p, r


def shd_tpr(W_e, W_t, thr=0.3):
    Wb = (np.abs(W_e) > thr)
    Tb = (np.abs(W_t) > 0)
    tp = int((Wb & Tb).sum()); fp = int((Wb & ~Tb).sum()); fn = int((~Wb & Tb).sum())
    shd = fp + fn
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fdr = fp / (tp + fp) if tp + fp else 0.0
    return shd, tpr, fdr
