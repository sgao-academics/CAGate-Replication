# -*- coding: utf-8 -*-
"""Thin adapter exposing run_fit.load(cancer, d) to the batch-3 drivers.

run_fit.py (repository root) is the end-to-end entry point that recomputes
data/pan_cancer/*.json from the raw UCSC-Xena expression matrices.  The drivers
that only need its expression loader import it through this two-argument shim.
"""
import os

from run_fit import load as _load

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TCGA_DIR = os.environ.get('TCGA_DATA_DIR', os.path.join(REPO_ROOT, 'tcga'))


def load(cancer, d):
    """Return (X, genes) for one cancer, top-d by variance, z-scored."""
    return _load(cancer, d, TCGA_DIR)
