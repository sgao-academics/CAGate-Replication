"""
CAGate replication -- one-click runner
======================================
Regenerates every figure of the manuscript from the pre-computed results in
``data/``.  No GPU, no network, no TCGA download: about half a minute on a
laptop.

Usage:
    python run_all.py

Output:
    figures/                 Fig1_Mechanism.pdf, Fig2_Pancancer.pdf,
                             Fig3_Validation.pdf, Fig4_Consensus.pdf
    figures_supplementary/   FigS1_HighDim.pdf, FigS2_Alpha.pdf,
                             FigS3_Panel.pdf, FigS4_Modules.pdf

Every generator is a self-contained matplotlib script that reads only from
``data/`` and writes only into the two figure directories, so a fresh clone
reproduces the published figures exactly.  The two style helpers (``figstyle``,
``palette``) and the corrected solver (``cagate``) are imported from this root.
"""
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))

# (script, output produced)
STEPS = [
    ('_fig1_gen.py',  'figures/Fig1_Mechanism.pdf'),
    ('_fig2_gen.py',  'figures/Fig2_Pancancer.pdf'),
    ('_fig3_gen.py',  'figures/Fig3_Validation.pdf'),
    ('_fig4_gen.py',  'figures/Fig4_Consensus.pdf'),
    ('_figS1_gen.py', 'figures_supplementary/FigS1_HighDim.pdf'),
    ('_figS2_gen.py', 'figures_supplementary/FigS2_Alpha.pdf'),
    ('_figS3_gen.py', 'figures_supplementary/FigS3_Panel.pdf'),
    ('_figS4_gen.py', 'figures_supplementary/FigS4_Modules.pdf'),
]


def run(script, expected):
    path = os.path.join(HERE, script)
    if not os.path.exists(path):
        print('  MISSING  %s' % script)
        return False
    t0 = time.time()
    r = subprocess.run([sys.executable, path], cwd=HERE, capture_output=True,
                       text=True, encoding='utf-8', errors='replace')
    dt = time.time() - t0
    for line in (r.stdout or '').strip().split('\n'):
        if line.strip():
            print('      ' + line.strip())
    if r.returncode != 0:
        print('  FAIL     %s (%d)' % (script, r.returncode))
        if r.stderr:
            print(r.stderr.strip()[-800:])
        return False
    ok = os.path.exists(os.path.join(HERE, expected))
    print('  %s  %s  (%.1fs)' % ('OK      ' if ok else 'NO FILE ', script, dt))
    return ok


def main():
    print('CAGate replication -- regenerating figures from data/')
    print('=' * 62)
    n_ok = 0
    for script, expected in STEPS:
        if run(script, expected):
            n_ok += 1
    print('=' * 62)
    print('%d of %d figures written' % (n_ok, len(STEPS)))
    return 0 if n_ok == len(STEPS) else 1


if __name__ == '__main__':
    sys.exit(main())
