"""
Fig1: CAGate Mechanism Diagram.
If pdflatex is available, compiles from fig1_mechanism.tex.
Otherwise, copies the pre-built Fig1_Mechanism.pdf from the package root.
"""
import os, subprocess, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

pdf_dst = os.path.join(FIG_DIR, 'Fig1_Mechanism.pdf')

# Try pdflatex first
try:
    result = subprocess.run(
        'pdflatex -interaction=nonstopmode -output-directory figures fig1_mechanism.tex',
        shell=True, cwd=HERE,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30
    )
    pdf_src = os.path.join(FIG_DIR, 'fig1_mechanism.pdf')
    if os.path.exists(pdf_src):
        if os.path.exists(pdf_dst):
            os.remove(pdf_dst)
        shutil.move(pdf_src, pdf_dst)
        for ext in ['.aux', '.log']:
            tmp = os.path.join(FIG_DIR, f'fig1_mechanism{ext}')
            if os.path.exists(tmp):
                os.remove(tmp)
        print(f'[SAVED] Fig1_Mechanism.pdf ({os.path.getsize(pdf_dst)//1024} KB) [built from TikZ]')
        sys.exit(0)
except Exception:
    pass

# Fallback: copy pre-built PDF
prebuilt = os.path.join(HERE, 'Fig1_Mechanism_prebuilt.pdf')
if os.path.exists(prebuilt):
    shutil.copy2(prebuilt, pdf_dst)
    print(f'[SAVED] Fig1_Mechanism.pdf ({os.path.getsize(pdf_dst)//1024} KB) [pre-built]')
else:
    print('TikZ compile failed and no pre-built PDF found.')
    print('Install pdflatex or ensure Fig1_Mechanism_prebuilt.pdf is in the package root.')
    sys.exit(1)
