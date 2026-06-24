# CAGate: Cluster-Aware Gating for Differentiable Causal Discovery in Cancer Genomics

> Shuaidong Gao, Chongqing Institute of Foreign Studies  
> Submitted to *Bioinformatics*

CAGate extends NOTEARS by embedding adaptive clustering into the augmented Lagrangian optimization loop. A MAD-aware gate weights per-cluster contributions so homogeneous subpopulations drive edge discovery. Across 33 TCGA cancer types, CAGate universally outperforms NOTEARS (mean Δ = +158 edges).

---

## Quick Start

```bash
git clone https://github.com/gsd3247186514-cell/CAGate-Replication.git
cd CAGate-Replication
pip install -r requirements.txt
python scripts/_gen_fig2_pancancer.py   # generates Fig2 from experimental data
```

All 10 figure-generation scripts are in `scripts/`. Each can be run independently.

---

## Repository Structure

```
CAGate-Replication/
├── 01_Manuscript.pdf                     Main paper (16 pages, 41 references)
├── 03_Supplementary_Information.pdf      Supplementary materials (6 figures, 9 tables, 17 pages)
├── scripts/                              Figure generation (10 scripts, standalone)
│   ├── _gen_fig1_mechanism.py            Fig1: CAGate mechanism diagram
│   ├── _gen_fig2_pancancer.py            Fig2: 33-cancer pan-cancer scatter
│   ├── _gen_fig3_highdim.py              Fig3: high-dimensional synthetic benchmark
│   ├── _gen_fig4_validation.py           Fig4: clinical database validation
│   └── _gen_figs[1-6]_*.py               FigS1-S6: supplementary figures
├── figures/                              4 main figures (PNG + PDF)
├── figures_supplementary/                6 supplementary figures (PNG + PDF)
├── cagate/                               Core algorithm (6 modules)
│   ├── canonical_cagate.py               CAGate main algorithm
│   ├── cluster_aware_loss.py             Cluster-weighted MSE loss
│   ├── notears_linear.py                 NOTEARS baseline
│   ├── notears_utils.py                  Augmented Lagrangian utilities
│   ├── download_tcga.py                  TCGA data download (UCSC Xena)
│   └── cagate_replication.py             End-to-end replication pipeline
├── data/                                 Experimental results
│   └── mega_33_full.json                 33-cancer benchmark (3 seeds per cancer)
├── results/validation/                   External validation data
│   ├── drug_targets.json                 CTD drug-target overlap
│   ├── cancer_drivers.json               ClinGen cancer driver overlap
│   ├── string.json                       STRING PPI overlap
│   ├── kegg_enrichment.json              KEGG pathway enrichment
│   ├── msigdb_enrichment.json            MSigDB pathway enrichment
│   ├── cancer_edge_counts.json           Edge counts per cancer
│   ├── trrust.json                       TRRUST TF-target overlap
│   └── biogrid.json                      BioGRID interaction overlap
├── references.bib                        All 41 references (BibTeX)
├── requirements.txt                      Python dependencies
├── README.md                             This file
└── LICENSE                               MIT License
```

---

## Requirements

- Python >= 3.9 (tested on 3.12)
- numpy >= 1.20, scipy >= 1.7, matplotlib >= 3.4
- No GPU required for figure reproduction

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Reproducing Figures

All figure scripts are self-contained and use relative paths. Run from the repository root:

```bash
# Main figures
python scripts/_gen_fig1_mechanism.py     # Mechanism diagram (no data dependency)
python scripts/_gen_fig2_pancancer.py     # 33-cancer pan-cancer scatter
python scripts/_gen_fig3_highdim.py       # Synthetic benchmark high-dim comparison
python scripts/_gen_fig4_validation.py    # Clinical validation (CTD/ClinGen/STRING)

# Supplementary figures
python scripts/_gen_figs1_flowchart.py    # S1: CAGate training flowchart
python scripts/_gen_figs2_prior.py        # S2: Prior knowledge comparison
python scripts/_gen_figs3_deepdive.py     # S3: Deep-dive analysis (6 panels)
python scripts/_gen_figs4_highdim_gain.py # S4: High-dimensional gain
python scripts/_gen_figs5_validation.py   # S5: KEGG pathway enrichment
python scripts/_gen_figs6_ranked_delta.py # S6: Ranked delta distribution
```

Expected runtime: ~5-10 seconds per figure. Output written to `figures/` and `figures_supplementary/`.

---

## Reproducing Full Experiments

To reproduce the complete 33-cancer benchmark:

```bash
python cagate/cagate_replication.py
```

**Hardware required:** GPU with >= 8 GB VRAM (tested on NVIDIA RTX 5060).  
**Expected runtime:** ~6-8 hours for all 33 cancers x 3 seeds.  
**Output:** `data/mega_33_full.json` (pre-computed checkpoint included in the repository).

Pre-computed checkpoints allow reviewers to verify figures without rerunning experiments.

---

## Data Sources

| Data | Source | Access |
|:--|:--|:--|
| TCGA expression (33 cancers) | [UCSC Xena](https://xenabrowser.net/) | Public, download via `cagate/download_tcga.py` |
| CTD drug-target interactions | [Comparative Toxicogenomics Database](http://ctdbase.org/) | Public |
| ClinGen cancer drivers | [Clinical Genome Resource](https://clinicalgenome.org/) | Public |
| STRING PPIs | [STRING v12](https://string-db.org/) | Public |
| KEGG pathways | [KEGG](https://www.genome.jp/kegg/) | Public |
| MSigDB gene sets | [MSigDB](https://www.gsea-msigdb.org/) | Public |

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{gao2026cagate,
  title   = {CAGate: Cluster-Aware Gating for Differentiable Causal Discovery in Cancer Genomics},
  author  = {Gao, Shuaidong},
  journal = {Bioinformatics},
  year    = {2026},
  note    = {Submitted}
}
```

---

## License

MIT License. See `LICENSE` file.

---

## Contact

**Shuaidong Gao**  
Chongqing Institute of Foreign Studies  
Qijiang Campus, Chongqing 401420, China  
Email: gsd3247186514@gmail.com  
ORCID: [0009-0004-5641-3581](https://orcid.org/0009-0004-5641-3581)  
GitHub: [CAGate-Replication](https://github.com/gsd3247186514-cell/CAGate-Replication)
