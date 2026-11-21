"""Build case_study_loan_reminders.pdf from report.md.

Usage:
    python make_figures.py        # figures/*.png from masked_results.json
    python build_report.py        # report.md -> pandoc -> LaTeX -> tectonic -> PDF

Needs pandoc and tectonic on the PATH (macOS: brew install pandoc tectonic).
"""
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
BUILD = HERE / "build"
PDF = HERE / "case_study_loan_reminders.pdf"


def body_latex(md: str) -> str:
    md = re.sub(r"^# .*\n", "", md, count=1)                    # title goes on the title page
    md = re.sub(r"^> .*(\n> .*)*\n", "", md, count=1, flags=re.M)  # the notice is in the title page box
    md = md.replace("[What was masked](#what-was-masked)", "What was masked")
    md = md.replace("[`../method/README.md`](../method/README.md)", "`method/README.md` in this repository")
    res = subprocess.run(
        ["pandoc", "-f", "markdown+pipe_tables+smart-auto_identifiers", "-t", "latex",
         "--wrap=preserve", "--shift-heading-level-by=-1"],
        input=md, capture_output=True, text=True, check=True)
    tex = res.stdout
    # figures: full text width, kept near their text
    tex = tex.replace("\\begin{figure}", "\\begin{figure}[H]")
    tex = re.sub(r"\\includegraphics(\[[^\]]*\])?\{", r"\\includegraphics[width=\\linewidth]{", tex)
    # each top-level section on a new page
    tex = re.sub(r"\\section\{", r"\\clearpage\n\\section{", tex)
    return tex


def main() -> None:
    md = (HERE / "report.md").read_text()
    pre = (HERE / "report_src" / "preamble.tex").read_text()
    fill = {
        "FONTSIZE": "10pt",
        "TITLE": "Payment reminders before the due date",
        "SUBTITLE": "A multi-arm uplift study on two randomized experiments",
        "AUDIENCE": "For business and technical readers",
        "META": "Real experiment at a consumer finance lender, masked for publication",
        "RUNHEAD": "Payment reminders: multi-arm uplift (masked)",
        "SECTIONBREAK": "\\clearpage",
        "BODY": body_latex(md),
    }
    for k, v in fill.items():
        pre = pre.replace(f"<<{k}>>", v)
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir()
    shutil.copytree(HERE / "figures", BUILD / "figures")
    (BUILD / "report.tex").write_text(pre)
    res = subprocess.run(["tectonic", "-X", "compile", "report.tex"], cwd=BUILD,
                         capture_output=True, text=True)
    if res.returncode:
        raise SystemExit(res.stdout[-3000:] + res.stderr[-3000:])
    warnings = [l for l in res.stderr.splitlines() if "warning" in l.lower()]
    shutil.copy(BUILD / "report.pdf", PDF)
    print(f"wrote {PDF.name}; {len(warnings)} warnings")
    for w in warnings[:20]:
        print(" ", w)


if __name__ == "__main__":
    main()
