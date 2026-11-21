"""Build the two case study reports (LaTeX) from the executed notebook's outputs.

    python build_case_study.py --execute     # run 03_case_study_beer.ipynb, then build both PDFs
    python build_case_study.py               # build both PDFs from the existing outputs/
    python build_case_study.py --only business
    python build_case_study.py --pages DIR   # also render every PDF page to PNG in DIR (visual check)

Outputs:
    case_study_business.pdf    for business readers: the decision, the money, the pilot, the risks
    case_study_technical.pdf   for data scientists: data, method, every result and check, Parts 1 to 14

Inputs (written by the notebook run):
    outputs/case_study_numbers.json   every number quoted in the text
    outputs/captions.json             caption and producing step of each figure and table
    outputs/figures/*.png, outputs/tables/*.csv
    03_case_study_beer.ipynb          cell ranges of each Part; code excerpts are checked against it

Pipeline: report_src/<name>.md.j2 (Jinja2 + Markdown) -> pandoc -> LaTeX body, wrapped in
report_src/preamble.tex -> tectonic -> PDF. Tables and figures are typeset here (booktabs), not by
pandoc, and numbered in order of appearance so the text can refer to them.

Needs pandoc and tectonic on the PATH (macOS: brew install pandoc tectonic). Tectonic downloads
the LaTeX packages it needs on the first run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import jinja2
import nbformat
import pandas as pd

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "03_case_study_beer.ipynb"
OUT = HERE / "outputs"
SRC = HERE / "report_src"
BUILD = OUT / "latex"

DOCS = {
    "business": dict(fontsize="11pt", title="Which trade action for which beer outlet?",
                     subtitle="What a per-outlet model of trade actions is worth, what is proven, and how to test it",
                     audience="Business report: for trade marketing and sales leadership",
                     sectionbreak=r"\needspace{0.3\textheight}\addvspace{1.2em}",
                     runhead="Beer trade actions: business report (synthetic data)"),
    "technical": dict(fontsize="10pt", title="Which trade action for which beer outlet?",
                      subtitle="A multi-arm uplift case study: effects per outlet, decisions under cost and capacity, "
                               "checked against the truth",
                      audience="Technical report: for data scientists and analysts",
                      sectionbreak=r"\needspace{0.3\textheight}\addvspace{1.6em}",
                      runhead="Beer trade actions: technical report (synthetic data)"),
}


# ── notebook ────────────────────────────────────────────────────────────────
def execute_notebook(timeout: int = 3600) -> None:
    from nbclient import NotebookClient
    nb = nbformat.read(NOTEBOOK, as_version=4)
    print(f"executing {NOTEBOOK.name} ...", flush=True)
    NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
    nbformat.write(nb, NOTEBOOK)
    print("notebook executed and saved", flush=True)


def part_cells(nb) -> dict:
    """{part number: (first cell, last cell)}, 1-based positions counting every cell."""
    starts = []
    for i, c in enumerate(nb.cells):
        m = re.match(r"## Part (\d+)\.", c.source) if c.cell_type == "markdown" else None
        if m:
            starts.append((int(m.group(1)), i))
    out = {}
    for j, (p, i) in enumerate(starts):
        end = starts[j + 1][1] - 1 if j + 1 < len(starts) else len(nb.cells) - 1
        out[p] = (i + 1, end + 1)
    return out


def check_excerpts(md_text: str, nb) -> list:
    """Every non-empty line of every ```python block must appear in a notebook code cell."""
    code_lines = {ln.strip() for c in nb.cells if c.cell_type == "code" for ln in c.source.splitlines()}
    missing = []
    for block in re.findall(r"```python\n(.*?)```", md_text, re.S):
        for ln in block.splitlines():
            s = ln.strip()
            if s and s != "..." and s not in code_lines:
                missing.append(s)
    return missing


# ── LaTeX helpers ───────────────────────────────────────────────────────────
_TEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "<": r"\textless{}",
        ">": r"\textgreater{}"}


def tex(s) -> str:
    return "".join(_TEX.get(ch, ch) for ch in str(s))


def fmt_cell(v) -> str:
    if isinstance(v, str):
        return "{}" + tex(v) if v.startswith("[") else tex(v)     # "[" after \\ would be read as a length
    if v is None or (not isinstance(v, (list, dict)) and pd.isna(v)):
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int) or (isinstance(v, float) and float(v).is_integer() and abs(v) >= 100):
        return f"{int(v):,}"
    if isinstance(v, float):
        if abs(v) >= 100:
            return f"{v:,.0f}"
        return f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 1 else f"{v:,.2f}".rstrip("0").rstrip(".")
    return tex(v)


def head_cell(name: str, numeric: bool, width: int = 16) -> str:
    """Header text broken into short lines so narrow numeric columns stay narrow."""
    lines, cur = [], ""
    for w in str(name).split(" "):
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    body = r" \\ ".join(tex(x) for x in lines)
    return rf"\thead[{'rb' if numeric else 'lb'}]{{{body}}}"


class Registry:
    """Numbers figures and tables in order of appearance and resolves references."""

    def __init__(self, known=None):
        self.known = dict(known or {})
        self.seen = {}
        self.count = {"fig": 0, "tab": 0}

    def number(self, key):
        kind = key.split(":")[0]
        if key not in self.seen:
            self.count[kind] += 1
            self.seen[key] = self.count[kind]
        return self.seen[key]

    def ref(self, key):
        n = self.known.get(key) or self.seen.get(key)
        if n is None:
            raise KeyError(f"reference to {key}, which is never shown")
        return ("Figure " if key.startswith("fig:") else "Table ") + str(n)


def make_env(numbers, captions, cells, reg: Registry, blocks: dict):
    """Jinja environment. table(), figure() and friends return a placeholder token; the LaTeX goes to `blocks`."""
    def token(latex: str, inline: bool = False) -> str:
        key = f"ZZBLOCK{len(blocks):04d}ZZ"
        blocks[key] = latex
        return key if inline else f"\n\n{key}\n\n"

    arm_names = numbers["arm_labels"]

    def table(name, cols=None, rename=None, rows=None, query=None, size="", max_rows=None, note="",
              caption=None, source=True, wrap=None):
        df = pd.read_csv(OUT / "tables" / f"{name}.csv", keep_default_na=True)
        df = df.convert_dtypes(convert_string=False, convert_boolean=True)
        if query:
            df = df.query(query)
        if rows is not None:
            df = df.iloc[rows]
        if max_rows:
            df = df.head(max_rows)
        if cols:
            df = df[cols]
        for c in ("arm", "chosen_arm", "true_best_arm"):              # arm codes -> names (0 -> control, ...)
            if c in df.columns and "label" not in df.columns and pd.api.types.is_integer_dtype(df[c]):
                df[c] = [f"{int(v)} {arm_names[str(int(v))]}" if pd.notna(v) else "" for v in df[c]]
        df = df.rename(columns=rename or {})
        key = f"tab:{name}"
        n = reg.number(key)
        cap = captions.get(key, {"caption": "", "step": ""})
        numeric = [pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]) for c in df.columns]
        spec, wrap = [], dict(wrap or {})
        for c, isnum in zip(df.columns, numeric):
            if c in wrap:
                spec.append(f">{{\\raggedright\\arraybackslash}}p{{{wrap[c]}}}")
            elif not isnum and df[c].astype(str).str.len().max() > 38:
                spec.append(">{\\raggedright\\arraybackslash}p{0.42\\linewidth}")
            else:
                spec.append("r" if isnum else "l")
        cells_txt = [[fmt_cell(v.item() if hasattr(v, "item") else v) for v in r] for r in df.itertuples(index=False)]
        # rough natural width in characters: widest cell or header line per column, paragraph columns count 30
        est = 0
        for j, c in enumerate(df.columns):
            if spec[j].startswith(">"):
                est += 30
                continue
            cell_w = max([len(re.sub(r"\\[a-z]+\{\}|\\", "", r[j])) for r in cells_txt] + [0])
            head_w = max(len(x) for x in re.split(r"(?<=.{16}) ", str(c))) if " " in str(c) else min(len(str(c)), 18)
            est += max(cell_w, head_w) + 2
        auto = r"\small" if est <= 88 else (r"\footnotesize" if est <= 104 else r"\scriptsize")
        font = {"small": r"\footnotesize", "tiny": r"\scriptsize"}.get(size, auto)
        if size == "small" and auto == r"\scriptsize":
            font = auto
        head = " & ".join(head_cell(c, isnum) for c, isnum in zip(df.columns, numeric)) + r" \\"
        body = "\n".join(" & ".join(r) + r" \\" for r in cells_txt)
        text = caption if caption is not None else cap["caption"]
        src = rf" {{\color{{muted}}\textit{{Source: {tex(cap['step'])}.}}}}" if source and cap.get("step") else ""
        nt = f" {tex(note)}" if note else ""
        ncol = len(spec)
        sep = {r"\small": "4pt", r"\footnotesize": "3pt"}.get(font, "2.5pt")
        caption_par = rf"\noindent{{\small\textbf{{Table {n}.}} {tex(text)}{nt}{src}\par}}"
        if len(cells_txt) <= 30:      # short table: caption and table kept together on one page
            latex = (rf"\par\begin{{minipage}}{{\linewidth}}{caption_par}\vspace{{0.3em}}" "\n"
                     rf"{{{font}\setlength{{\tabcolsep}}{{{sep}}}\begin{{tabular}}{{@{{}}{''.join(spec)}@{{}}}}" "\n"
                     rf"\toprule {head} \midrule" "\n" f"{body}\n" r"\bottomrule\end{tabular}\par}"
                     r"\end{minipage}\par\vspace{0.9em}")
        else:                         # long table: may break across pages, header repeated
            latex = (rf"\par\needspace{{8\baselineskip}}{caption_par}\nopagebreak[4]" "\n"
                     rf"\begingroup{font}\setlength{{\tabcolsep}}{{{sep}}}"
                     rf"\setlength{{\LTpre}}{{0.2em}}\setlength{{\LTpost}}{{0.8em}}" "\n"
                     rf"\begin{{longtable}}[l]{{@{{}}{''.join(spec)}@{{}}}}" "\n"
                     rf"\toprule {head} \midrule \endfirsthead" "\n"
                     rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\small\textit{{Table {n}, continued}}}} \\ \toprule {head} \midrule \endhead" "\n"
                     rf"\bottomrule \endlastfoot" "\n" f"{body}\n" r"\end{longtable}\endgroup")
        return token(latex)

    def figure(name, width="100%", note="", wide=False, caption=None, source=True, height=None):
        key = f"fig:{name}"
        n = reg.number(key)
        cap = captions.get(key, {"caption": "", "step": ""})
        text = caption if caption is not None else cap["caption"]
        src = rf" {{\color{{muted}}\textit{{Source: {tex(cap['step'])}.}}}}" if source and cap.get("step") else ""
        nt = f" {tex(note)}" if note else ""
        frac = float(str(width).rstrip("%")) / 100
        hgt = height or r"0.80\textheight"
        size = rf"width={frac:.2f}\linewidth,height={hgt},keepaspectratio"
        if wide:
            size = r"width=\linewidth,height=0.66\textheight,keepaspectratio"
        img = f"../figures/{name}.png"                     # relative to outputs/latex
        latex = (r"\begin{figure}[H]\centering" "\n"
                 rf"\includegraphics[{size}]{{{img}}}" "\n"
                 rf"\caption*{{\textbf{{Figure {n}.}} {tex(text)}{nt}{src}}}" "\n" r"\end{figure}")
        return token(latex)

    def raw(latex):
        return token(latex)

    def tval(name, col, **match):
        """One cell of an exported table, picked by column equality matches."""
        df = pd.read_csv(OUT / "tables" / f"{name}.csv")
        for k, v in match.items():
            df = df[df[k] == v]
        if len(df) != 1:
            raise ValueError(f"tval({name}, {col}, {match}) matched {len(df)} rows")
        v = df[col].iloc[0]
        return v.item() if hasattr(v, "item") else v

    def tcount(name, query=None):
        """Number of rows of an exported table, optionally after a pandas query."""
        df = pd.read_csv(OUT / "tables" / f"{name}.csv")
        return len(df.query(query)) if query else len(df)

    def nbref(part):
        a, b = cells[part]
        return token(rf"\nbref{{Notebook Part {part}, cells {a} to {b}.}}")

    env = jinja2.Environment(undefined=jinja2.StrictUndefined, autoescape=False,
                             loader=jinja2.FileSystemLoader(str(SRC)))
    env.filters["vnd"] = lambda x: f"{x:,.0f}"
    env.filters["pct"] = lambda x, d=1: f"{100 * x:.{d}f}%"
    env.filters["f"] = lambda x, d=2: f"{x:,.{d}f}"
    env.filters["mvnd"] = lambda x: f"{x / 1e6:,.0f}"
    env.filters["r3"] = lambda x: round(x, -3)                      # "about" numbers in narrative text
    env.globals["phi"] = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))   # standard normal CDF (pilot power)
    env.globals.update(table=table, figure=figure, nbref=nbref, ref=reg.ref, n=numbers, tval=tval, tcount=tcount,
                       landscape_begin=lambda: raw(r"\begin{landscape}"),
                       landscape_end=lambda: raw(r"\end{landscape}"),
                       box_begin=lambda title="": raw(
                           rf"\begin{{keybox}}[title={{{tex(title)}}}]" if title else r"\begin{keybox}"),
                       box_end=lambda: raw(r"\end{keybox}"),
                       warn_begin=lambda: raw(r"\begin{warnbox}"),
                       warn_end=lambda: raw(r"\end{warnbox}"),
                       newpage=lambda: raw(r"\clearpage"),
                       need=lambda lines=12: raw(rf"\needspace{{{lines}\baselineskip}}"))
    return env


def render_markdown(doc: str, numbers, captions, cells):
    reg1 = Registry()
    make_env(numbers, captions, cells, reg1, {}).get_template(f"{doc}.md.j2").render()       # pass 1: numbering
    reg, blocks = Registry(reg1.seen), {}
    text = make_env(numbers, captions, cells, reg, blocks).get_template(f"{doc}.md.j2").render()
    return text, blocks, reg


def to_latex(md_text: str, blocks: dict) -> str:
    body = subprocess.run(
        ["pandoc", "-f", "markdown+raw_tex+pipe_tables+smart-implicit_figures-auto_identifiers",
         "-t", "latex", "--listings", "--wrap=preserve", "--shift-heading-level-by=-1"],
        input=md_text, capture_output=True, text=True, check=True).stdout
    # inline code: plain \texttt (\lstinline breaks inside tables and arguments); \allowbreak after _ . (
    def inline(m):
        code = tex(re.sub(r"\\([_%#{}&$])", r"\1", m.group(2)))
        code = re.sub(r"(\\_|\.|\()", lambda x: x.group(1) + r"\allowbreak{}", code)
        return rf"\texttt{{{code}}}"
    body = re.sub(r"\\passthrough\{\\lstinline(.)(.*?)\1\}", inline, body)

    # short pandoc tables: tabular instead of longtable (longtable can leave its head alone at a page top)
    def short_table(m):
        spec, head, rows = m.group(1), m.group(2), m.group(3)
        lines = [r for r in re.split(r"(?<=\\\\)\n", rows) if r.strip()]
        chunks = [lines[k:k + 8] for k in range(0, len(lines), 8)] if len(lines) > 30 else [lines]
        out = []
        for k, ch in enumerate(chunks):
            top = (r"\toprule" + head + r"\midrule" + "\n") if k == 0 else ""
            bottom = (r"\bottomrule" + "\n") if k == len(chunks) - 1 else ""
            glue = "" if k == 0 else r"\vspace{-\parskip}"
            out.append(rf"{glue}\par\noindent\begin{{tabular}}{{{spec}}}" + "\n" + top + "\n".join(ch) + "\n" + bottom
                       + r"\end{tabular}\par")
        return "\n".join(out)
    body = re.sub(r"\{\\def\\LTcaptype\{none\} % do not increment counter\n\\begin\{longtable\}\[\]\{(.*?)\}\n"
                  r"\\toprule\\noalign\{\}(.*?)\\midrule\\noalign\{\}\n\\endhead\n\\bottomrule\\noalign\{\}\n"
                  r"\\endlastfoot\n(.*?)\\end\{longtable\}\n\}", short_table, body, flags=re.S)
    for key, latex in blocks.items():
        if key not in body:
            raise RuntimeError(f"placeholder {key} lost in pandoc output")
        body = body.replace(key, latex)
    return body


def compile_pdf(doc: str, body: str, numbers) -> Path:
    meta = DOCS[doc]
    pre = (SRC / "preamble.tex").read_text(encoding="utf-8")
    if doc == "business":
        info = rf"Companion: technical report \textperiodcentered{{}} built {dt.date.today().isoformat()}"
    else:
        info = (f"Modelling library version {tex(numbers['environment']['auto_uplift_multi_arms'])} (private)"
                rf" \textperiodcentered{{}} notebook \texttt{{03\_case\_study\_beer.ipynb}}"
                rf" \textperiodcentered{{}} built {dt.date.today().isoformat()}")
    for k, v in {"FONTSIZE": meta["fontsize"], "TITLE": tex(meta["title"]), "SUBTITLE": tex(meta["subtitle"]),
                 "AUDIENCE": tex(meta["audience"]), "META": info, "RUNHEAD": tex(meta["runhead"]), "SECTIONBREAK": meta["sectionbreak"], "BODY": body}.items():
        pre = pre.replace(f"<<{k}>>", v)
    BUILD.mkdir(parents=True, exist_ok=True)
    src = BUILD / f"case_study_{doc}.tex"
    src.write_text(pre, encoding="utf-8")
    res = subprocess.run(["tectonic", "-X", "compile", "--keep-logs", src.name], cwd=BUILD,
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"tectonic failed for {doc}:\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}")
    pdf = HERE / f"case_study_{doc}.pdf"
    shutil.copyfile(BUILD / f"case_study_{doc}.pdf", pdf)
    log = (BUILD / f"case_study_{doc}.log").read_text(encoding="utf-8", errors="replace")
    for ln in re.findall(r"^(?:Overfull \\hbox \((?:\d{2,}|[5-9])\.\d+pt.*|Missing character.*|.*undefined.*)$",
                         log, re.M)[:15]:
        print("  latex:", ln.strip())
    return pdf


def business_figures() -> None:
    """Plain-language redraws for the business report, from tables the notebook exported (no new numbers)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = pd.read_csv(OUT / "tables" / "promo_vs_share_observed.csv")

    def label(b):
        if b == "missing":
            return "not\nsurveyed"
        lo, hi = (float(x) for x in b.strip("[]()").split(","))
        return f"{lo:.0%}\nto {hi:.0%}"
    x = range(len(t))
    fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=200)
    ax.axhline(2.0, color="#444444", ls="--", lw=1, label="Break-even: 2 cases")
    ax.axhline(0.0, color="#999999", lw=0.6)
    ax.errorbar(x, t.actual, yerr=[t.actual - t.actual_ci_low, t.actual_ci_high - t.actual], fmt="o", color="#1F4E79",
                ecolor="#8EA9C8", elinewidth=2, capsize=0, ms=5, label="Measured in the data (95% range)")
    ax.plot(list(x)[:-1], t.effect[:-1], color="#C2410C", lw=2.2, marker="s", ms=4, label="Model prediction")
    ax.plot(list(x)[-1:], t.effect[-1:], color="#C2410C", marker="s", ms=4, ls="none")
    ax.set_xticks(list(x))
    ax.set_xticklabels([label(b) for b in t.bin], fontsize=9.5)
    ax.set_xlabel("Our brand's share of the outlet's beer volume (survey estimate)", fontsize=10.5)
    ax.set_ylabel("Extra cases per month from a promo", fontsize=10.5)
    ax.tick_params(axis="y", labelsize=9.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "business_promo_by_share.png")
    plt.close(fig)


def build(only=None, pages_dir=None) -> list:
    numbers = json.loads((OUT / "case_study_numbers.json").read_text())
    captions = json.loads((OUT / "captions.json").read_text())
    nb = nbformat.read(NOTEBOOK, as_version=4)
    cells = part_cells(nb)
    pdfs, shown = [], set()
    business_figures()
    for doc in DOCS:
        if only and doc != only:
            continue
        text, blocks, reg = render_markdown(doc, numbers, captions, cells)
        missing = check_excerpts(text, nb)
        if missing:
            raise SystemExit(f"{doc}: code excerpts not found in the notebook:\n  " + "\n  ".join(missing))
        if "TODO" in text or "\u2014" in text:
            raise SystemExit(f"{doc}: TODO or em dash left in the text")
        shown |= set(reg.seen)
        (BUILD).mkdir(parents=True, exist_ok=True)
        (BUILD / f"case_study_{doc}.md").write_text(text, encoding="utf-8")
        pdf = compile_pdf(doc, to_latex(text, blocks), numbers)
        print(f"wrote {pdf.name} ({pdf.stat().st_size / 1e6:.1f} MB)")
        pdfs.append(pdf)
        if pages_dir:
            render_pages(pdf, Path(pages_dir) / doc)
    if not only:
        unused = sorted(set(captions) - shown)
        if unused:
            print("note: exported but in neither PDF:", ", ".join(unused))
    return pdfs


def render_pages(pdf: Path, out: Path, scale: float = 1.4) -> None:
    import pypdfium2 as pdfium
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("page_*.png"):
        f.unlink()
    doc = pdfium.PdfDocument(str(pdf))
    for i in range(len(doc)):
        doc[i].render(scale=scale).to_pil().save(out / f"page_{i + 1:02d}.png")
    print(f"rendered {len(doc)} pages to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true", help="run the notebook first")
    ap.add_argument("--only", choices=list(DOCS), help="build one report only")
    ap.add_argument("--pages", default=None, help="render every page to PNG in this folder")
    a = ap.parse_args()
    if a.execute:
        execute_notebook()
    build(a.only, a.pages)
