"""
CAGate replication -- one-click runner
======================================
Regenerates every figure of the manuscript from the pre-computed results in
``data/``.  No GPU, no network, no TCGA download: about 20 seconds on a laptop.

Usage:
    python run_all.py

Output:
    figures/                 Fig1_Mechanism.pdf, Fig2_Pancancer.pdf, Fig3_Validation.pdf
    figures_supplementary/   FigS1_Supplementary.pdf, FigS3_HighDim.pdf

Figure 1 is a TikZ diagram; if ``pdflatex`` is absent the pre-built
``Fig1_Mechanism_prebuilt.pdf`` is copied instead and the run still succeeds.
"""
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))

# (script, output produced)
STEPS = [
    ('gen_fig1_mechanism.py', 'figures/Fig1_Mechanism.pdf'),
    ('gen_fig2_precision.py', 'figures/Fig2_Pancancer.pdf'),
    ('gen_fig3_scope.py',     'figures/Fig3_Validation.pdf'),
    ('gen_figs1_prior.py',    'figures_supplementary/FigS1_Supplementary.pdf'),
    ('gen_figs2_highdim.py',  'figures_supplementary/FigS3_HighDim.pdf'),
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
