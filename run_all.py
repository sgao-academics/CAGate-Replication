"""
CAGate Replication — One-Click Runner
======================================
Reproduces all 6 figures (Fig1-4, FigS1-2) from pre-computed results.
No internet required. No GPU required. ~30 seconds on a modern laptop.

Usage:
    python run_all.py
    python run_all.py --fig1-only     # TikZ mechanism diagram only
    python run_all.py --skip-fig1     # Skip TikZ (if pdflatex not installed)

Dependencies:
    pip install -r requirements.txt
    pdflatex (for Fig1 only; optional — pre-built Fig1 included)

Output:
    figures/          — Fig1-4 (manuscript figures)
    figures_supplementary/  — FigS1-2 (supplementary figures)
"""
import os, sys, subprocess, shutil, time

HERE = os.path.dirname(os.path.abspath(__file__))
SKIP_FIG1 = '--skip-fig1' in sys.argv
FIG1_ONLY = '--fig1-only' in sys.argv

def header(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")

def run_script(name):
    """Run a Python script and report success/failure."""
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        print(f"  SKIP: {name} not found")
        return False
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, path],
        cwd=HERE,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='ignore'
    )
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"  OK  {name} ({elapsed:.1f}s)")
        # Print stdout safely (handle Windows GBK encoding)
        stdout_text = result.stdout or ''
        if stdout_text.strip():
            for line in stdout_text.strip().split('\n'):
                try:
                    print(f"      {line}")
                except UnicodeEncodeError:
                    print(f"      [line with special characters]")
        return True
    else:
        print(f"  FAIL {name} (exit {result.returncode})")
        return False

def verify_files():
    """Check all expected outputs exist."""
    expected = {
        'figures/Fig1_Mechanism.pdf': 'Fig1: Mechanism diagram',
        'figures/Fig2_Pancancer.pdf': 'Fig2: Pan-cancer performance',
        'figures/Fig3_HighDim.pdf': 'Fig3: High-dimensional rescue',
        'figures/Fig4_Validation.pdf': 'Fig4: External validation',
        'figures_supplementary/FigS1_Supplementary.pdf': 'FigS1: Prior comparison',
        'figures_supplementary/FigS2_Supplementary.pdf': 'FigS2: Per-cancer delta',
    }
    ok, missing = 0, []
    for path, desc in expected.items():
        fp = os.path.join(HERE, path)
        if os.path.exists(fp):
            sz = os.path.getsize(fp)
            ok += 1
            print(f"  [OK] {path} ({sz//1024} KB) — {desc}")
        else:
            missing.append((path, desc))
            print(f"  [MISSING] {path} — {desc}")
    return ok, missing

# ── Main ──
header("CAGate Replication Pipeline")
print(f"  Working directory: {HERE}")
print(f"  Python: {sys.version.split()[0]}")

# Check dependencies
try:
    import numpy; import scipy; import matplotlib; import pandas
    import sklearn; import networkx; import seaborn
    print("  Core dependencies: OK")
except ImportError as e:
    print(f"  MISSING DEPENDENCY: {e}")
    print("  Run: pip install -r requirements.txt")
    sys.exit(1)

if FIG1_ONLY:
    header("Fig1: Mechanism Diagram (TikZ)")
    run_script('gen_fig1_mechanism.py')
else:
    scripts = [
        ('gen_fig2_pancancer.py', 'Fig2: Pan-cancer Performance'),
        ('gen_fig3_highdim.py', 'Fig3: High-Dimensional Rescue'),
        ('gen_fig4_validation.py', 'Fig4: External Validation'),
        ('gen_figs1.py', 'FigS1: Prior Comparison'),
        ('gen_figs2.py', 'FigS2: Per-Cancer Delta'),
    ]
    if not SKIP_FIG1:
        scripts.insert(0, ('gen_fig1_mechanism.py', 'Fig1: Mechanism Diagram (TikZ)'))

    failures = 0
    for name, desc in scripts:
        header(desc)
        if not run_script(name):
            failures += 1

    header("Verification")
    ok, missing = verify_files()
    print(f"\n  Generated: {ok}/6 figures")
    if missing:
        print(f"  Missing: {len(missing)}")
        for path, desc in missing:
            print(f"    - {path}: {desc}")
    if failures:
        print(f"\n  {failures} script(s) failed. See output above for details.")
    else:
        print(f"\n  All scripts passed. Replication complete.")

print(f"\n{'='*60}")
print("  Output locations:")
print(f"    {os.path.join(HERE, 'figures')}/")
print(f"    {os.path.join(HERE, 'figures_supplementary')}/")
print(f"{'='*60}")
