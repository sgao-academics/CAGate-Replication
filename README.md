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
no GPU, no network, no TCGA download, about 20 seconds. Output goes to
`figures/` (Fig. 1–3) and `figures_supplementary/` (Fig. S1–S2).

## Layout

| Path | Contents |
|:--|:--|
| `run_all.py` | One-click regeneration of every figure |
| `cagate.py` | The CAGate solver: doubled-variable L-BFGS-B with an augmented Lagrangian and a residual-contrast cluster gate |
| `notears_linear.py`, `notears_utils.py` | The NOTEARS baseline, run to convergence, used throughout |
| `gen_fig1_mechanism.py` | Figure 1 — mechanism diagram (TikZ, with a pre-built fallback) |
| `gen_fig2_precision.py` | Figure 2 — edge precision across 33 TCGA cancers |
| `gen_fig3_scope.py` | Figure 3 — mechanism and scope |
| `gen_figs1_prior.py` | Supplementary Figure S1 — prior comparison (TCGA-BRCA, *d* = 300) |
| `gen_figs2_highdim.py` | Supplementary Figure S2 — high-dimensional analysis |
| `_prior_experiment.py` | The experiment behind Supplementary Figure S1 |
| `download_tcga.py` | TCGA RNA-Seq download (UCSC Xena) |
| `data/` | Pre-computed results that every figure is drawn from |
| `environment.yml` | Conda specification of the experimental environment |
| `SHA256SUMS.txt` | SHA-256 manifest of this package |

The file `figures_supplementary/FigS3_HighDim.pdf` is Supplementary Figure
**S2** of the manuscript; it keeps the name used by the manuscript's
`\includegraphics`.

## What the paper claims — and what it does not

- **Precision, not recall.** The gate returns *fewer* edges than the converged
  NOTEARS baseline (50.3 versus 63.7 on average over the 33 cancers) while a
  larger share of them are supported by an independent protein–protein
  interaction database.
- **Scope: cluster heterogeneity.** On homogeneous Erdős–Rényi graphs, where
  there is no structure to exploit, the change in structural Hamming distance is
  never negative. The gain is confined to samples that carry genuine cluster
  structure.
- **Orientation is not recovered.** The directed (TRRUST) analysis supports
  *enrichment* — gate 4.64 % versus NOTEARS 3.28 % of TF-source edges forming a
  curated regulatory pair, *P* = 0.040 — but forward support (2.49 %) is not
  distinguishable from the reversed direction (2.28 %, *P* = 1.00). No claim of
  recovered directionality is made.
- **NOTEARS does not collapse.** Run to convergence, NOTEARS tracks the truth at
  every swept dimension on synthetic data and rises monotonically on real
  expression data; it does not collapse at *d* = 200.

An earlier version of this package argued the opposite on the last two points
(an edge-count advantage and a "collapse" of NOTEARS at high dimension). Both
came from an under-converged reference solver and are withdrawn; the
implementation shipped here is the corrected one.

## Data provenance

| `data/` | Backs |
|:--|:--|
| `evidence.json` | Figures 2 and 3; the aggregate over the 33 cancers |
| `dirval_results.json` | Figure 2(d) and Supplementary Table S6 |
| `real_dim/brca_d{50,100,150,200}.json` | Supplementary Figure S2(b) and Supplementary Table S3 |
| `pan_cancer/*.json` | The 33 per-cancer runs behind Figure 2 and Supplementary Table S5 |
| `tf_enriched/*.json` | The transcription-factor-enriched runs behind Figure 2(d) and Supplementary Table S6 |
| `synthetic_dimension_sweep/*.json` | Supplementary Figure S2(a) and Supplementary Table S2 |
| `cluster_sweep/*.json` | Figure 3(a) |
| `prior_ckpt.json` | Supplementary Figure S1 |

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
