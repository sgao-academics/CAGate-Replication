# CAGate: Cluster-Aware Gating Improves the Precision of Causal Discovery in Heterogeneous Cancer Transcriptomes

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![Preprint](https://img.shields.io/badge/SSRN-10.2139%2Fssrn.7164939-orange)](https://doi.org/10.2139/ssrn.7164939)

Replication package for *CAGate: Cluster-Aware Gating Improves the Precision of
Causal Discovery in Heterogeneous Cancer Transcriptomes* — Shuaidong Gao
(Chongqing Institute of Foreign Studies).

CAGate adds a **cluster-aware gate** to NOTEARS-style differentiable causal
discovery. On 33 TCGA cancer types at *d* = 100 the gate recovers **fewer**
edges than the ungated arms, but a **larger fraction** of them are independently
supported: 29.9 % of gated edges versus 25.4 % for NOTEARS against STRING
(combined score ≥ 700), a paired difference of **+4.43 pp** that is positive in
**32 of 33** cancers (Wilcoxon signed-rank *P* = 1.6 × 10⁻⁸).

## Quick start

```bash
git clone https://github.com/sgao-academics/CAGate-Replication.git
cd CAGate-Replication
pip install -r requirements.txt
python run_all.py
```

`run_all.py` regenerates every figure from the pre-computed results in `data/` —
no GPU, no network, no TCGA download, under a minute. Output goes to
`figures/` (Fig. 1–4) and `figures_supplementary/` (Fig. S1–S5).

## Layout

| Path | Contents |
|:--|:--|
| `run_all.py` | One-click regeneration of every figure |
| `cagate.py` | The CAGate solver: doubled-variable L-BFGS-B with an augmented Lagrangian and a residual-contrast cluster gate |
| `notears_linear.py`, `notears_utils.py` | The NOTEARS baseline, run to convergence, used throughout |
| `_fig1_gen.py` | Figure 1 — mechanism: residual dispersion → gate → cluster weights |
| `_fig2_gen.py` | Figure 2 — edge precision across 33 TCGA cancers |
| `_fig3_gen.py` | Figure 3 — mechanism: cluster heterogeneity, weight-threshold retention, and insensitivity to α |
| `_fig4_gen.py` | Figure 4 — cross-cancer consensus: the support ladder, sign stability, the multi-source forest, and the degree × strength control |
| `_figS1_gen.py` | Supplementary Figure S1 — high-dimensional behaviour |
| `_figS2_gen.py` | Supplementary Figure S2 — behaviour and robustness of the gate |
| `_figS3_gen.py` | Supplementary Figure S3 — what each gene panel can measure, and what it can be tested against |
| `_figS4_gen.py` | Supplementary Figure S4 — what the consensus edges are made of |
| `_figS5_gen.py` | Supplementary Figure S5 — boundary conditions: homogeneous graphs, and re-partitioned labels |
| `figstyle.py`, `palette.py` | Figure-style helpers (shared with the manuscript's other figures) |
| `download_tcga.py` | TCGA RNA-Seq download (UCSC Xena) |
| `data/` | Pre-computed results that every figure is drawn from |
| `environment.yml` | Conda specification of the experimental environment |
| `SHA256SUMS.txt` | SHA-256 manifest of this package |

Every generator writes the exact file names the manuscript's `\includegraphics`
expects, so `run_all.py` reproduces the published figures with no renaming step.

## What the paper claims — and what it does not

- **Precision, not recall.** The gate returns *fewer* edges than the converged
  NOTEARS baseline (50.3 versus 63.7 on average over the 33 cancers) while a
  larger share of them are supported by an independent protein–protein
  interaction database.
- **What the gate removes.** Pooled over the 33 cohorts, the gate keeps 48 % of
  the weakest absolute-weight stratum but 83 % of the strongest, so it prunes the
  weakly supported tail while retaining the well-supported edges (Figure 3b).
- **Scope: cluster heterogeneity.** On homogeneous Erdős–Rényi graphs there is
  no structure to exploit and the gate adds no structural accuracy
  (Supplementary Figure S5a); the gain is confined to samples that carry genuine
  cluster structure.
- **Orientation is not recovered.** The directed (TRRUST) analysis supports
  *enrichment* — gate 4.64 % versus NOTEARS 3.28 % of TF-source edges forming a
  curated regulatory pair, *P* = 0.040 — but forward support (2.49 %) is not
  distinguishable from the reversed direction (2.28 %, *P* = 1.00). No claim of
  recovered directionality is made.
- **NOTEARS does not collapse.** Run to convergence, NOTEARS tracks the truth at
  every swept dimension on synthetic data and rises monotonically on real
  expression data; it does not collapse at *d* = 200.

All comparisons in this package use a NOTEARS solver run to convergence.

## Data provenance

| `data/` | Backs |
|:--|:--|
| `evidence.json` | Figures 2 and 3(a), Supplementary Figures S1(a) and S5(a) — the aggregate over the 33 cancers (external-support tests, cluster-heterogeneity runs, homogeneous ER sweep) |
| `dirval_results.json` | Figure 2(d) and Supplementary Table S6 |
| `real_dim/brca_d{50,100,150,200}.json` | Supplementary Figure S1(b) and Supplementary Table S3 |
| `synthetic_dimension_sweep/*.json` | Supplementary Table S2 — the raw ER sweep also carried in `evidence.json` |
| `pan_cancer/*.json` | Supplementary Table S5 — the raw 33 per-cancer runs also carried in `evidence.json` |
| `tf_enriched/*.json` | Figure 2(d) and Supplementary Table S6 |
| `cluster_sweep/*.json` | Figure 3(a) — paired cluster-heterogeneity runs |
| `cluster_rf/*.json` | Supplementary Figure S5(b) — fixed labels versus re-partitioned labels |
| `mega33_results/*.json` | Figure 3(b) and Supplementary Figure S2(a) — the 33 per-cancer edge counts at each weight threshold |
| `alpha_sweep.json` | Figure 3(c), Supplementary Figure S2(b) and Supplementary Table S4 — the *α* sensitivity sweep |
| `fig4_data.json` | Figure 4 — every panel |
| `figS34_data.json` | Supplementary Figures S3 and S4 |

TCGA RNA-Seq data are public and are not redistributed here;
`download_tcga.py` fetches them from UCSC Xena. All experiments use fixed random
seeds.

## Environment

`environment.yml` reproduces the experimental environment (Python 3.12, PyTorch
2.11+cu128). Regenerating the figures needs only `requirements.txt`.

## Integrity

```bash
sha256sum -c SHA256SUMS.txt
```

## Citation

```bibtex
@article{gao2026cagate,
  title  = {CAGate: Cluster-Aware Gating Improves the Precision of Causal Discovery in Heterogeneous Cancer Transcriptomes},
  author = {Gao, Shuaidong},
  year   = {2026},
  doi    = {10.2139/ssrn.7164939}
}
```

## Patent

The CAGate method is the subject of patent application CNIPA 202611098494.0
(filed 23 July 2026).

## See also

[`causalscale`](https://github.com/sgao-academics/causalscale) is a standalone
Python package that bundles this implementation together with other
differentiable causal-discovery methods and diagnostics.

## License

MIT — see [LICENSE](LICENSE).
