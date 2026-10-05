# -*- coding: utf-8 -*-
"""Regime of the gate: does the residual-contrast gate survive when clusters
differ in STRUCTURE (cluster-specific edges) rather than only in noise scale?

The paper's clustered benchmark (SM3) varies only the noise scale c_k, with the
same DAG W in every cluster.  Reviewer A calls this circular: the gate assumes
"large residual = noise", and the benchmark makes that true by construction.

Here we add three regimes, all with the same solver and the same gate:
  V0  noise-only heterogeneity          (reproduces the paper's benchmark)
  V1  structural heterogeneity, SYMMETRIC : every cluster gets n_extra own edges
  V2  structural heterogeneity, ASYMMETRIC: only clusters {0,1} get own edges
                                            (this is the paper's own motivation:
                                             "a relation holding in one subtype
                                             is attenuated by the subpopulations
                                             in which it is absent")

Target graph = the UNION of all cluster graphs (every relation that is real
somewhere).  Arms: NOTEARS (pooled) / base (clustered, uniform weights) / gate.

Everything ASCII on stdout; full record to JSON.
"""
import io, os, sys, json, time
import numpy as np

# Paths are resolved relative to this file so the script runs from a fresh
# clone. External inputs (TCGA panels, STRING v12 raw files) are located via
# the environment variables documented in the repository README.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
sys.path.insert(0, REPO_ROOT)
from cagate import solve_cagate_lbfgsb, f1, gen_cluster           # noqa: E402
from notears_linear import notears_linear                          # noqa: E402

D, K, SP = 15, 5, 100
L1 = 0.1
ALPHA = 0.5
SEEDS = [0, 1, 2, 3, 4]


def _backbone(rng, d, s0):
    Wt = np.zeros((d, d)); e = 0
    while e < s0:
        i, j = rng.randint(0, d, 2)
        if i < j and Wt[i, j] == 0:
            Wt[i, j] = rng.uniform(0.5, 2.0) * rng.choice([-1, 1]); e += 1
    return Wt


def gen_struct(d=D, nc=K, sp=SP, seed=0, n_extra=0, carriers="all"):
    """Shared backbone DAG + cluster-specific extra edges (all DAG-safe)."""
    rng = np.random.RandomState(seed)
    Wt = _backbone(rng, d, 2 * d)
    extra = []
    for k in range(nc):
        Ek = np.zeros((d, d))
        if n_extra > 0 and (carriers == "all" or k in carriers):
            e, guard = 0, 0
            while e < n_extra and guard < 20000:
                i, j = rng.randint(0, d, 2)
                if i < j and Wt[i, j] == 0 and Ek[i, j] == 0:
                    Ek[i, j] = rng.uniform(0.5, 2.0) * rng.choice([-1, 1]); e += 1
                guard += 1
        extra.append(Ek)
    perm = rng.permutation(d)
    Ws = [(Wt + Ek)[perm][:, perm] for Ek in extra]
    Wb = Wt[perm][:, perm]
    U = np.zeros((d, d))
    for Wk in Ws:
        U = np.maximum(U, (np.abs(Wk) > 0).astype(float))
    X = np.zeros((nc * sp, d)); cid = np.zeros(nc * sp, int)
    for k in range(nc):
        Xk = rng.randn(sp, d) @ np.linalg.inv(np.eye(d) - Ws[k])
        X[k * sp:(k + 1) * sp] = Xk
        cid[k * sp:(k + 1) * sp] = k
    return X, cid, Wb, U


def run_arms(X, cid, truth):
    nts = notears_linear(X, lambda1=L1, loss_type="l2")
    base = solve_cagate_lbfgsb(X, cid=cid, use_gate=False, l1=L1)["W"]
    gt = solve_cagate_lbfgsb(X, cid=cid, use_gate=True, gate_alpha=ALPHA, l1=L1)
    return dict(notears=f1(nts, truth)[0],
                base=f1(base, truth)[0],
                gate=f1(gt["W"], truth)[0],
                gates=gt["gates"])


def block(label, X, cid, truth, out, seed, extra=None):
    t0 = time.time()
    r = run_arms(X, cid, truth)
    rec = dict(model=label, seed=seed, f1_notears=r["notears"],
               f1_base=r["base"], f1_gate=r["gate"],
               d_gate_base=r["gate"] - r["base"],
               d_gate_notears=r["gate"] - r["notears"],
               secs=round(time.time() - t0, 1))
    if extra:
        rec.update(extra)
    out.append(rec)
    print("  %-34s s=%d  base=%.3f gate=%.3f  gate-base=%+.3f" %
          (label, seed, r["base"], r["gate"], rec["d_gate_base"]))
    return r


def main():
    out = []
    print("=" * 78)
    print("A-M1  structural heterogeneity vs the gate  (d=%d, K=%d, %d/cluster)"
          % (D, K, SP))
    print("=" * 78)

    # ---------- V0: the paper's own benchmark (noise-only) ----------
    print("\n[V0] noise-only heterogeneity (paper's SM3 generator)")
    for s in SEEDS:
        X, Wt, cid = gen_cluster(d=D, nc=K, sp=SP, seed=s)
        block("V0 noise-only (vs backbone)", X, cid, Wt, out, s)

    # ---------- V1/V2: structural heterogeneity ----------
    for rho in (0.25, 0.5, 1.0):
        ne = int(round(rho * 2 * D))
        print("\n[V1] SYMMETRIC structural heterogeneity  (rho=%.2f, %d extra edges/cluster)"
              % (rho, ne))
        for s in SEEDS:
            X, cid, Wb, U = gen_struct(seed=s, n_extra=ne, carriers="all")
            r = block("V1 sym rho=%.2f" % rho, X, cid, U, out, s,
                      {"rho": rho, "truth": "union"})
            out[-1]["gamma_extracarrying"] = float(np.mean(r["gates"]))

        print("\n[V2] ASYMMETRIC structural heterogeneity (rho=%.2f, clusters {0,1} only)"
              % rho)
        for s in SEEDS:
            X, cid, Wb, U = gen_struct(seed=s, n_extra=ne, carriers=(0, 1))
            r = block("V2 asym rho=%.2f" % rho, X, cid, U, out, s,
                      {"rho": rho, "truth": "union"})
            g = r["gates"]; m = cid
            out[-1]["gamma_carrying"] = float(np.mean(g[(m == 0) | (m == 1)]))
            out[-1]["gamma_other"] = float(np.mean(g[(m != 0) & (m != 1)]))

    # ---------- summary ----------
    print("\n" + "=" * 78)
    print("%-34s %8s %8s %8s %8s" % ("model", "F1_nt", "F1_base", "F1_gate", "g-b"))
    print("-" * 78)
    keys = []
    for r in out:
        if r["model"] not in keys:
            keys.append(r["model"])
    for k in keys:
        rs = [r for r in out if r["model"] == k]
        f = lambda p: np.mean([r[p] for r in rs])                      # noqa: E731
        print("%-34s %8.3f %8.3f %8.3f %+8.3f" %
              (k, f("f1_notears"), f("f1_base"), f("f1_gate"), f("d_gate_base")))

    gs = [r for r in out if "gamma_carrying" in r]
    if gs:
        print("\nper-cluster gate weight (V2 asymmetric):")
        for k in dict.fromkeys(r["model"] for r in gs):
            rs = [r for r in gs if r["model"] == k]
            print("  %-20s carrying clusters gamma=%.3f | others gamma=%.3f" %
                  (k, np.mean([r["gamma_carrying"] for r in rs]),
                   np.mean([r["gamma_other"] for r in rs])))

    p = os.path.join(DATA_DIR, 'hetero', 'hetero_results.json')
    with io.open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote", p)


if __name__ == "__main__":
    main()
