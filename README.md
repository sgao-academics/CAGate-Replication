# CAGate: Cluster-Aware Gating and Cross-Cancer Consensus Improve the Precision of Causal Discovery in Heterogeneous Cancer Transcriptomes

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![Preprint](https://img.shields.io/badge/SSRN-10.2139%2Fssrn.7164939-orange)](https://doi.org/10.2139/ssrn.7164939)

Replication package for *CAGate: Cluster-Aware Gating and Cross-Cancer Consensus
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

## Reproducibility levels

| Level | Command | What it reproduces |
|:--|:--|:--|
| 1 — figures | `python run_all.py` | Every figure, from `data/*.json`; under a minute, no network, no GPU |
| 2 — results | `python run_fit.py --data-dir <tcga> [CHOL ...]` | Re-fits the three arms from the raw expression matrices and rewrites `data/pan_cancer/*.json`; needs the TCGA matrices and ~2–9 min per cancer on CPU |
| 3 — raw data | `python download_tcga.py` | Fetches TCGA RNA-Seq from UCSC Xena |
| 3b — notes SN21–SN23 | `python scripts/<driver>.py` | Re-runs the analyses added after the preprint (Supplementary Notes SN21–SN23); the synthetic ones need only this package, the database ones also need the STRING v12 raw files |

Level 2 closes the gap left by level 1, which regenerates the figures but not the numbers they plot. The re-fit is exact for a fixed solver stack; the one requirement is **single-threaded BLAS** (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, which `run_fit.py` sets itself). Multithreaded BLAS changes the floating-point reduction order inside L-BFGS-B and can shift the flattest arm (base) by a few edges; single-threaded, the refit reproduces **all 33 cancer types exactly** — sample size, gene panel, and the base / gate / NOTEARS edge counts all match the released `data/pan_cancer/*.json` (33 / 33). The manuscript quotes the counts from the released checkpoints, which these reproduce.

## Layout

| Path | Contents |
|:--|:--|
| `run_all.py` | One-click regeneration of every figure (level 1 below) |
| `run_fit.py` | Re-fits the three arms from the raw TCGA matrices and rewrites `data/pan_cancer/*.json` (level 2 below) |
| `scripts/` | The drivers behind Supplementary Notes SN21–SN23 — structural heterogeneity versus dispersion, the published GENIE3 baseline with its degree-matched null, and the STRING threshold sweep — with their helper modules and the vendored GENIE3 implementation |
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
| `hetero/*.json` | Supplementary Note SN21 and Table S12 — dispersion versus structure |
| `genie3/{synth,real,vim}/` | Supplementary Note SN22 — the published tree-ensemble baseline, and the score matrices behind its degree-matched null |
| `hubnull/hubnull.json` | Supplementary Note SN22 — the stub-matched (degree-preserving) null |
| `string_threshold/*.json` | Supplementary Note SN23 — external support at combined score ≥ 400 / 700 / 900 |
| `mega33_w/*` | The three arms' weight matrices — the common input of SN22 and SN23 |

TCGA RNA-Seq data are public and are not redistributed here;
`download_tcga.py` fetches them from UCSC Xena. All experiments use fixed random
seeds.

## Notes SN21–SN23 (added after the preprint)

| Note | Driver | Reproducible from |
|:--|:--|:--|
| SN21 — dispersion versus structure | `scripts/_A_M1_hetero.py`, `scripts/_A_M1b_gate_scale.py` | this package alone (synthetic) |
| SN22 — published baseline (GENIE3) | `scripts/_b3_genie3.py`, `scripts/_b3_genie3_vim.py`, `scripts/_b3_hubnull.py` | `data/mega33_w/` plus the STRING v12 raw files |
| SN23 — STRING threshold sweep | `scripts/_b3_stringthr.py` | `data/mega33_w/`, `data/genie3/vim/` plus the STRING v12 raw files |

The synthetic note (SN21) runs end to end from a fresh clone with no external
input — `python scripts/_A_M1_hetero.py`. The two database notes locate their
external input through environment variables, because neither STRING nor TCGA is
redistributed here:

| Variable | Points at | Used by |
|:--|:--|:--|
| `STRING_DATA_DIR` | the directory holding the STRING v12 `9606.protein.links.v12.0.txt.gz` and its alias file | SN22, SN23 |
| `TCGA_DATA_DIR` | a directory of `TCGA_<CANCER>_HiSeqV2.tsv` files | the expression loader shared with `run_fit.py` |
| `STRING_CACHE_DIR` | optional scratch directory for the parsed STRING pools (defaults to `data/_cache/`) | SN22, SN23 |

GENIE3 is vendored verbatim from the author's repository
(`scripts/genie3_vendor/GENIE3.py`, `github.com/vahuynh/GENIE3`), so the baseline
runs without a network dependency. The driver scripts keep their original working
names on purpose, so that each one can be lined up with the note it produces.

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
  title  = {CAGate: Cluster-Aware Gating and Cross-Cancer Consensus Improve the Precision of Causal Discovery in Heterogeneous Cancer Transcriptomes},
  author = {Gao, Shuaidong},
  year   = {2026},
  doi    = {10.2139/ssrn.7164939}
}
```

## Preprint

An earlier version is on SSRN ([10.2139/ssrn.7164939](https://doi.org/10.2139/ssrn.7164939)).
It corresponds to a pre-submission draft; the present package is the version
submitted to the *Journal of Bioinformatics and Computational Biology* and
supersedes it. The signed-edge and cross-cancer-consensus analyses were added
after the preprint.

## Patent

The CAGate method is the subject of patent application CNIPA 202611098494.0
(filed 23 July 2026).

## See also

[`causalscale`](https://github.com/sgao-academics/causalscale) is a standalone
Python package that bundles this implementation together with other
differentiable causal-discovery methods and diagnostics.

## License

MIT — see [LICENSE](LICENSE).

The code is released under the MIT Licence. The CAGate *method* is the subject of
patent application CNIPA 202611098494.0 (and a corresponding US provisional
application); the MIT grant covers the code as published and does not by itself
grant a patent licence. Academic and other non-commercial research use of the
method as described in the paper is permitted; commercial use of the patented
method may require a separate licence.
