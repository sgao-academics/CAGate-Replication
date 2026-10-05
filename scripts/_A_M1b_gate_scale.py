# -*- coding: utf-8 -*-
"""A-M1b: is the gate's real effect a CONTRAST effect, or a global loss-rescaling
artefact?

The logistic gamma = 1/(1+exp(-a(sigma_bar - sigma))) maps the median cluster to
0.5, not 1.  So the gated objective scales the data term by ~0.5 while the L1
term lambda1|W| is untouched -- i.e. the gate silently multiplies the effective
sparsity penalty by ~2.  If the "precision gain" is mostly that, then
mean-normalising gamma should remove it.

We re-run V0 (noise-only, where the paper reports a gain) with
  (i)  the gate as published          gamma_k
  (ii) the gate mean-normalised       gamma_k / mean(gamma)
and compare against the ungated base.
"""
import io, os, sys, json
import numpy as np

# Paths are resolved relative to this file so the script runs from a fresh
# clone. External inputs (TCGA panels, STRING v12 raw files) are located via
# the environment variables documented in the repository README.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
sys.path.insert(0, REPO_ROOT)
import cagate                                                        # noqa: E402
from cagate import solve_cagate_lbfgsb, f1, gen_cluster               # noqa: E402

D, K, SP, L1, ALPHA = 15, 5, 100, 0.1, 0.5
SEEDS = [0, 1, 2, 3, 4]
_ORIG = cagate.cluster_gate_np


def _norm_gate(res, cid, alpha=0.5):
    g, info = _ORIG(res, cid, alpha)
    m = float(np.mean(g))
    return (g / m if m > 0 else g), info


def solve(X, cid, mode):
    if mode == "published":
        cagate.cluster_gate_np = _ORIG
    else:
        cagate.cluster_gate_np = _norm_gate
    r = solve_cagate_lbfgsb(X, cid=cid, use_gate=True, gate_alpha=ALPHA, l1=L1)
    cagate.cluster_gate_np = _ORIG
    return r


rows = []
print("=" * 74)
print("V0 noise-only (paper's benchmark)  d=%d K=%d  n=%d" % (D, K, K * SP))
print("%-6s %10s %10s %10s %10s" % ("seed", "base_F1", "gate_F1", "gateNorm_F1", "edges b/g/gn"))
print("-" * 74)
for s in SEEDS:
    X, Wt, cid = gen_cluster(d=D, nc=K, sp=SP, seed=s)
    base = solve_cagate_lbfgsb(X, cid=cid, use_gate=False, l1=L1)
    gp = solve(X, cid, "published")
    gn = solve(X, cid, "norm")
    row = dict(seed=s,
               f1_base=f1(base["W"], Wt)[0],
               f1_gate=f1(gp["W"], Wt)[0],
               f1_gatenorm=f1(gn["W"], Wt)[0],
               e_base=base["n_edges"], e_gate=gp["n_edges"], e_gatenorm=gn["n_edges"],
               gmean=float(np.mean(gp["gates"])))
    rows.append(row)
    print("%-6d %10.3f %10.3f %10.3f        %3d/%3d/%3d   mean_gamma=%.3f" %
          (s, row["f1_base"], row["f1_gate"], row["f1_gatenorm"],
           row["e_base"], row["e_gate"], row["e_gatenorm"], row["gmean"]))

m = lambda k: float(np.mean([r[k] for r in rows]))                    # noqa: E731
print("-" * 74)
print("%-6s %10.3f %10.3f %10.3f        %3d/%3d/%3d" %
      ("MEAN", m("f1_base"), m("f1_gate"), m("f1_gatenorm"),
       m("e_base"), m("e_gate"), m("e_gatenorm")))
print("\ngate - base        = %+.3f" % (m("f1_gate") - m("f1_base")))
print("gateNorm - base    = %+.3f" % (m("f1_gatenorm") - m("f1_base")))
print("mean gamma (published gate) = %.3f" % m("gmean"))

with io.open(os.path.join(DATA_DIR, 'hetero', 'gate_scale_results.json'), 'w', encoding='utf-8') as fh:
    json.dump(rows, fh, indent=1)
print("wrote _A_M1b_gate_scale.json")
