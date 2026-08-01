# CAGate: Cluster-Aware Gating Rescues Differentiable Causal Discovery

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![SSRN](https://img.shields.io/badge/SSRN-10.2139%2Fssrn.7164939-orange)](https://doi.org/10.2139/ssrn.7164939)
[![CausalScale](https://img.shields.io/badge/CausalScale-37%20stars%209%20forks-2ea44f)](https://github.com/sgao-academics/causalscale)

**Replication package** for the paper *CAGate: Cluster-Aware Gating Rescues Differentiable Causal Discovery in Cancer Genomics* (Shuaidong Gao, 2026).

## What is CAGate?

CAGate extends NOTEARS with adaptive cluster-aware gating that addresses the gradient-dilution problem in high-dimensional transcriptomic data. A MAD-aware gate weights per-cluster contributions so that structure-rich subpopulations drive edge discovery while noise-dominated subsets are suppressed.

**Key results**: Across 33 TCGA cancer types (68 configurations), CAGate outperforms NOTEARS in all 33 cancers (mean +158 edges). At d=200, CAGate recovers 550+ edges where NOTEARS finds zero.

## Quick Start

```bash
# 1. Clone
git clone https://github.com/sgao-academics/CAGate-Replication.git
cd CAGate-Replication

# 2. Install dependencies
pip install -r requirements.txt

# 3. Reproduce all 6 figures (~10 seconds)
python run_all.py
```

Output: `figures/` (Fig1-4) + `figures_supplementary/` (FigS1-2). No GPU. No internet. No TCGA data download required.

## Repository Structure

```
.
├── run_all.py                  # One-click reproduction
├── gen_fig1_mechanism.py       # Fig1: Mechanism diagram (TikZ)
├── gen_fig2_pancancer.py       # Fig2: Pan-cancer performance
├── gen_fig3_highdim.py         # Fig3: High-dimensional rescue
├── gen_fig4_validation.py      # Fig4: External validation
├── gen_figs1.py                # FigS1: Prior comparison
├── gen_figs2.py                # FigS2: Per-cancer delta
├── fig1_mechanism.tex          # TikZ source for Fig1
├── Fig1_Mechanism_prebuilt.pdf # Pre-built Fig1 (pdflatex fallback)
├── cagate_delta_data.json      # Pre-computed CAGate vs NOTEARS deltas
├── _cross_results.json         # Pre-computed cross-method benchmarks
├── _cross_ckpt.json            # Cross-method checkpoint
├── _cross_ges_ckpt.json        # GES checkpoint
├── _prior_ckpt.json            # Prior comparison checkpoint
├── cagate_replication.py       # Reference CAGate implementation
├── canonical_cagate.py         # Canonical CAGate algorithm
├── cluster_aware_loss.py       # Cluster-aware loss function
├── notears_linear.py           # NOTEARS linear SEM
├── notears_utils.py            # NOTEARS utilities
├── download_tcga.py            # TCGA data download (UCSC Xena)
├── _cross_method_bg.py         # Cross-method experiment (optional re-run)
├── _cross_golem_bg.py          # GOLEM experiment (optional re-run)
├── _prior_experiment.py        # Prior comparison (optional re-run)
├── requirements.txt            # Python dependencies
├── references.bib              # Bibliography
├── LICENSE                     # MIT
└── README.md
```

## Figure-to-Script Mapping

| Figure | Script | Data Source |
|:--|:--|:--|
| **Fig1** — Mechanism diagram | `gen_fig1_mechanism.py` | TikZ / pre-built PDF |
| **Fig2** — Pan-cancer performance | `gen_fig2_pancancer.py` | `cagate_delta_data.json` + `_cross_results.json` |
| **Fig3** — High-dimensional rescue | `gen_fig3_highdim.py` | `_cross_results.json` |
| **Fig4** — External validation | `gen_fig4_validation.py` | Hardcoded validation results |
| **FigS1** — Prior comparison | `gen_figs1.py` | Hardcoded BRCA results |
| **FigS2** — Per-cancer delta | `gen_figs2.py` | `cagate_delta_data.json` |

## Re-Running From Scratch (Optional)

The scripts `_cross_method_bg.py`, `_cross_golem_bg.py`, and `_prior_experiment.py` can re-run all experiments from raw TCGA data. This requires:

1. Download TCGA data via `download_tcga.py` (outputs to `data/`)
2. ~10 hours on a modern CPU/GPU
3. Additional dependency: `causallearn` (for GES)

These are **not required** for figure reproduction — all figures use pre-computed checkpoints.

## Requirements

- Python 3.9+
- Packages: `numpy scipy matplotlib torch scikit-learn pandas networkx seaborn SciencePlots`
- pdflatex (optional — Fig1 uses pre-built fallback if unavailable)
- No GPU required

## Citation

```bibtex
@article{gao2026cagate,
  title={CAGate: Cluster-Aware Gating Rescues Differentiable Causal Discovery in Cancer Genomics},
  author={Gao, Shuaidong},
  year={2026},
  doi={10.2139/ssrn.7164939}
}
```

## Related

- [causalscale](https://github.com/sgao-academics/causalscale) — Unified causal discovery engine integrating CAGate, NOTEARS, GOLEM, DAGMA, and related diagnostics
- [Causal Transformer](https://github.com/sgao-academics/causalscale) — Scaling gradient-based causal discovery to 500+ variables
- Patent: China National IP Administration, application 202611098494.0 (filed July 23, 2026)

## License

MIT. See [LICENSE](LICENSE).
