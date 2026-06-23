# CAGate: Cluster-Aware Gating for Differentiable Causal Discovery

Replication package for the CAGate paper.

## Quick Start
```bash
pip install -r requirements.txt
python scripts/_gen_fig2_pancancer.py  # regenerates Fig2
python scripts/_gen_figs5_validation.py  # regenerates FigS5
```

All scripts are self-contained — read from `data/` and `results/validation/`, write to `figures/`.

## Structure
```
scripts/  — 10 figure generation scripts (one per figure)
data/     — mega_33_full.json (33 TCGA cancer benchmark)
results/validation/ — External validation databases
cagate/   — Algorithm implementation
figures/  — Main paper figures
figures_supplementary/ — Supplementary figures
```

## Requirements
Python 3.8+, numpy, scipy, matplotlib
