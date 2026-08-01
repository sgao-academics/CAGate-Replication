# CAGate: Cluster-Aware Gating Rescues Differentiable Causal Discovery

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![SSRN](https://img.shields.io/badge/SSRN-10.2139%2Fssrn.7164939-orange)](https://doi.org/10.2139/ssrn.7164939)
[![CausalScale](https://img.shields.io/badge/causalscale-37%E2%98%85-2ea44f)](https://github.com/sgao-academics/causalscale)

Replication package for *CAGate: Cluster-Aware Gating Rescues Differentiable Causal Discovery in Cancer Genomics* — Shuaidong Gao (2026), submitted to *Bioinformatics*.

CAGate extends NOTEARS with adaptive cluster-aware gating. Across 33 TCGA cancers, CAGate outperforms NOTEARS in all 33 (mean +158 edges). At d=200, CAGate recovers 550+ edges where NOTEARS finds zero.

> **🔧 Production users:** The full CAGate implementation — plus NOTEARS, GOLEM, DAGMA, Causal Transformer, LowRankGNN, and 12+ diagnostic tools — lives in **[causalscale](https://github.com/sgao-academics/causalscale)** (37⭐). `pip install causalscale` for the complete differentiable causal discovery toolkit. This repo contains the minimal replication materials for the Bioinformatics paper.

## Quick Start

```bash
git clone https://github.com/sgao-academics/CAGate-Replication.git
cd CAGate-Replication
pip install -r requirements.txt
python run_all.py        # 6 figures, ~10 seconds
```

No GPU. No internet. No TCGA download. Outputs to `figures/` and `figures_supplementary/`.

## Figures

| Figure | Script | Needs |
|:--|:--|:--|
| Fig1 — Mechanism | `gen_fig1_mechanism.py` | pre-built PDF fallback |
| Fig2 — Pan-cancer | `gen_fig2_pancancer.py` | `cagate_delta_data.json` |
| Fig3 — High-dim rescue | `gen_fig3_highdim.py` | `_cross_results.json` |
| Fig4 — Validation | `gen_fig4_validation.py` | self-contained |
| FigS1 — Prior comparison | `gen_figs1.py` | self-contained |
| FigS2 — Per-cancer delta | `gen_figs2.py` | `cagate_delta_data.json` |

## Re-Running Experiments (Optional)

`_cross_method_bg.py`, `_cross_golem_bg.py`, and `_prior_experiment.py` can re-run from raw TCGA data (~10 hours, requires `causallearn`). Not needed for figure reproduction.

## Citation

```bibtex
@article{gao2026cagate,
  title={CAGate: Cluster-Aware Gating Rescues Differentiable Causal Discovery in Cancer Genomics},
  author={Gao, Shuaidong},
  year={2026},
  doi={10.2139/ssrn.7164939}
}
```

Patent: CNIPA application 202611098494.0 (filed July 23, 2026).

MIT License.
