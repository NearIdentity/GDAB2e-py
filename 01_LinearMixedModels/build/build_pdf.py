"""Render the executed notebooks into the companion PDF (HTML -> Chromium), with corrections in the margin.
Run with a Python that has: nbformat, markdown, latex2mathml, pygments, playwright."""
import html, pathlib, re, subprocess, sys
import markdown, nbformat
from latex2mathml.converter import convert as tex2mml
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer, SASLexer, SLexer, TextLexer
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from corrections import CORRECTIONS, NOTES

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
OUT = ROOT / "GDAB2e-py-01_LinearMixedModels_PythonCompanion.pdf"
FMT = HtmlFormatter(nowrap=True)
LEX = {"SAS": SASLexer(), "R": SLexer(), "ASReml": TextLexer()}

PARTS = [  # (TOC title, notebook stem, kind)
    ("1. Mixed Models Compared to Traditional ANOVA", "01_anova_vs_mixed_models", "main"),
    ("2. Mixed-Model Analysis; Hypothesis Tests; BLUE and BLUP", "02_mixed_models_balanced", "main"),
    ("3. Unbalanced Data: ANOVA versus Mixed Models", "03_unbalanced_data", "main"),
    ("4. Mixed Models in a Nutshell", "04_theory_lactation", "main"),
    ("5. Appendix of the Chapter: Matrices with NumPy", "05_matrix_algebra", "main"),
    ("Appendix A. Data Sources and Setup", "00_data_sources", "app"),
    ("Appendix B. Choosing a Python Mixed-Model Engine", "06_appendix_B_alternatives", "app"),
    ("Appendix C. Custom Code: Inventory, Unit Tests and Benchmarks", "07_appendix_C_tests_benchmarks", "app"),
    ("Appendix C.6 Source Listings of lmmtools", None, "listing"),
    ("Appendix D. Register of Corrections and Notes", None, "register"),
    ("References", None, "refs"),
]
FIRST_H1 = {"00_data_sources": "Appendix A. Data Sources and Setup"}


def tex(s, display):
    m = tex2mml(s.strip(), display="block" if display else "inline")
    m = re.sub(r"<mi>&#x(1D4[0-3][0-9A-F]);</mi>", lambda k: f'<mi mathvariant="normal" style="font-weight:bold">'
               f'{chr(ord("A") + int(k.group(1), 16) - 0x1D400) if int(k.group(1), 16) - 0x1D400 < 26 else chr(ord("a") + int(k.group(1), 16) - 0x1D41A)}</mi>', m)
    m = m.replace('mathvariant="bold-italic"', 'style="font-weight:bold;font-style:italic"')
    return m.replace('mathvariant="bold"', 'mathvariant="normal" style="font-weight:bold"')


SHOWN = set()          # margin notes already placed in the current section


def margin(cid):
    if cid in SHOWN:
        return ""
    SHOWN.add(cid)
    if cid.startswith("C"):
        c = CORRECTIONS[cid]
        return (f'<span class="mnote corr-note"><b>{cid}</b> <i>{html.escape(c["where"])}.</i> '
                f'Draft: {html.escape(c["original"])}. <b>Corrected:</b> {html.escape(c["corrected"])}.</span>')
    n = NOTES[cid]
    return f'<span class="mnote note-note"><b>{cid}</b> {html.escape(n["text"])}</span>'


def md_html(text):
    store = []
    def keep(s):
        store.append(s); return f"\x00{len(store)-1}\x00"
    # corrections / notes first (their text may contain math)
    def corr(m):
        cid, inner = m.group(1), m.group(2)
        return keep(margin(cid)) + keep('<span class="corr">') + inner + keep(f'<sup class="cid">{cid}</sup></span>')
    text = re.sub(r"\{\{(C\d+)\|(.+?)\}\}", corr, text, flags=re.S)
    text = re.sub(r"\{\{(N\d+)\}\}", lambda m: keep(margin(m.group(1))) + keep(f'<sup class="nid">{m.group(1)}</sup>'), text)
    text = re.sub(r"\$\$(.+?)\$\$", lambda m: keep(f'<span class="dmath">{tex(m.group(1), True)}</span>'), text, flags=re.S)
    text = re.sub(r"(?<![\\$])\$(?!\$)(.+?)(?<!\\)\$", lambda m: keep(tex(m.group(1), False)), text)
    out = markdown.markdown(text, extensions=["tables", "fenced_code"])
    for _ in range(3):
        out = re.sub("\x00(\\d+)\x00", lambda m: store[int(m.group(1))], out)
    return out


def code_box(src, lexer, label, cls):
    return f'<div class="code {cls}"><div class="label">{label}</div><pre>{highlight(src, lexer, FMT)}</pre></div>'


def outputs(cell):
    parts = []
    for o in cell.get("outputs", []):
        t = o.get("output_type")
        if t == "stream" and o.get("name") == "stdout":
            txt = html.escape(o["text"].rstrip())
            txt = re.sub(r"^(PASS)(  .*)$", r'<span class="pass">\1</span>\2', txt, flags=re.M)
            parts.append(f"<pre class='out'>{txt}</pre>")
        elif t in ("execute_result", "display_data"):
            d = o.get("data", {})
            if "image/png" in d:
                parts.append(f"<div class='fig'><img src='data:image/png;base64,{d['image/png']}'/></div>")
            elif "text/html" in d and "<table" in d["text/html"]:
                parts.append(f"<div class='df'>{d['text/html']}</div>")
            elif "text/plain" in d:
                parts.append(f"<pre class='out'>{html.escape(d['text/plain'])}</pre>")
    return "".join(parts)


def nb_html(stem, title):
    nb = nbformat.read(NB / f"{stem}.ipynb", 4)
    SHOWN.clear()
    body, first = [], True
    for c in nb.cells:
        if c.cell_type == "markdown":
            m = re.match(r"\*\*Original (\S+) code \(from the chapter\)(?: — (.*?))?\*\*\n\n```[^\n]*\n(.*)\n```\s*$", c.source, re.S)
            if m:
                lab = f"Original {m.group(1)} (chapter)" + (f" — {m.group(2)}" if m.group(2) else "")
                body.append(code_box(m.group(3), LEX.get(m.group(1), TextLexer()), lab, "orig"))
                continue
            src = c.source.split("\n\n> **Correction")[0].split("\n\n> **Note")[0]     # notebook-only footers
            src = re.sub(r"<mark>(.+?)</mark><sup>\[(C\d+)\]</sup>", lambda k: "{{" + k.group(2) + "|" + k.group(1) + "}}", src, flags=re.S)
            src = re.sub(r"<sup>\[(N\d+)\]</sup>", lambda k: "{{" + k.group(1) + "}}", src)
            if first and stem in FIRST_H1:
                src = re.sub(r"^# .*", f"# {FIRST_H1[stem]}", src, count=1)
            first = False
            body.append(md_html(src))
        elif c.cell_type == "code":
            tags = c.metadata.get("tags", [])
            if "nopdf" in tags:
                continue
            if "imports" in tags:
                body.append(code_box(c.source, PythonLexer(), "Python modules and imports for this section", "imports"))
            else:
                body.append(code_box(c.source, PythonLexer(), "Python", "py"))
            body.append(outputs(c))
    return f"<section class='chapter' id='{stem}'>" + "\n".join(body) + "</section>"


def listing_html():
    files = ["src/lmmtools/reml.py", "src/lmmtools/anova.py", "src/lmmtools/mme.py", "src/lmmtools/design.py",
             "src/lmmtools/data.py", "src/lmmtools/report.py", "benchmarks/bench_reml.py"]
    parts = ["<section class='chapter'><h1>Appendix C.6 Source Listings of lmmtools</h1>"
             "<p>Complete source of the custom package (<code>src/lmmtools</code>) and the benchmark script, as executed for this document. "
             "C.1 = <code>reml.py</code>, C.2 = <code>anova.py</code>, C.3 = <code>mme.py</code>. The unit tests are in <code>tests/</code>.</p>"]
    for f in files:
        src = (ROOT / f).read_text()
        parts.append(f"<h3>{f} <span class='loc'>({len(src.splitlines())} lines)</span></h3>" + code_box(src, PythonLexer(), f, "py listing"))
    return "".join(parts) + "</section>"


def register_html():
    rows = "".join(f"<tr><td><b>{k}</b></td><td>{html.escape(v['where'])}</td><td>{html.escape(v['original'])}</td>"
                   f"<td>{html.escape(v['corrected'])}</td><td>{html.escape(v['why'])}</td><td>{html.escape(v['verified_by'])}</td></tr>"
                   for k, v in CORRECTIONS.items())
    notes = "".join(f"<tr><td><b>{k}</b></td><td>{html.escape(v['where'])}</td><td colspan='4'>{html.escape(v['text'])}</td></tr>"
                    for k, v in NOTES.items())
    return ("<section class='chapter wide'><h1>Appendix D. Register of Corrections and Notes</h1>"
            "<p>Corrections (C) fix errors in the March 2025 draft; notes (N) clarify behaviour that differs between software. "
            "In the text each corrected passage is <span class='corr'>highlighted<sup class='cid'>C#</sup></span> and explained in the margin. "
            "Where a correction is numerical, the cited unit test or notebook check verifies it.</p>"
            "<table class='reg'><tr><th>ID</th><th>Where (draft)</th><th>Draft says</th><th>Corrected</th><th>Why</th><th>Verified by</th></tr>"
            f"{rows}{notes}</table></section>")


def refs_html():
    return "<section class='chapter'>" + md_html((ROOT / "build" / "references.md").read_text()) + "</section>"


CSS = """
@page { size: Letter; margin: 0.7in 0.5in 0.8in 0.7in; }
body { font-family: 'DejaVu Serif', Georgia, serif; font-size: 9.6pt; line-height: 1.42; color: #1d1d1f; padding-right: 1.95in; }
h1 { font-family: 'DejaVu Sans', sans-serif; font-size: 17pt; color: #123a63; border-bottom: 2px solid #123a63; padding-bottom: 4px; margin-top: 0; }
h2 { font-family: 'DejaVu Sans', sans-serif; font-size: 12.5pt; color: #123a63; margin-top: 16px; break-after: avoid; }
h3 { font-family: 'DejaVu Sans', sans-serif; font-size: 10.5pt; color: #2b4f75; margin-top: 12px; break-after: avoid; }
.chapter { break-before: page; }
p, li { text-align: justify; }
table { border-collapse: collapse; margin: 8px auto; font-size: 8.4pt; }
th, td { border: 1px solid #b9c3cf; padding: 2px 6px; vertical-align: top; }
th { background: #e8eef5; }
.code { margin: 7px 0 3px 0; }
.code .label { font-family: 'DejaVu Sans', sans-serif; font-size: 7.2pt; font-weight: bold; letter-spacing: .04em; text-transform: uppercase;
  padding: 2px 8px; border-radius: 4px 4px 0 0; display: block; width: max-content; break-after: avoid; }
.code pre { margin: 0; padding: 6px 9px; font-size: 7.7pt; line-height: 1.3; white-space: pre-wrap; word-break: break-word;
  font-family: 'DejaVu Sans Mono', monospace; border-radius: 0 4px 4px 4px; orphans: 4; widows: 3; }
.orig .label { background: #e3e3e3; color: #444; } .orig pre { background: #f2f2f2; border: 1px solid #dcdcdc; }
.py .label { background: #123a63; color: #fff; } .py pre { background: #f4f8fc; border: 1px solid #c9d8e8; }
.imports .label { background: #1a7f37; color: #fff; } .imports pre { background: #eef8f0; border: 1px solid #b7dfc1; border-left: 4px solid #1a7f37; }
.listing pre { font-size: 6.9pt; }
pre.out { font-family: 'DejaVu Sans Mono', monospace; font-size: 7.3pt; line-height: 1.25; background: #fff; border-left: 3px solid #9fb6cc;
  margin: 2px 0 9px 0; padding: 3px 9px; white-space: pre-wrap; word-break: break-word; }
.pass { color: #1a7f37; font-weight: bold; }
.fig { text-align: center; margin: 4px 0 10px; } .fig img { max-width: 88%; max-height: 3.6in; }
.df table { font-size: 7.3pt; font-family: 'DejaVu Sans Mono', monospace; }
.dmath { display: block; margin: 7px 0; text-align: center; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.2pt; background: #f1f3f5; padding: 0 2px; border-radius: 2px; }
.corr { background: #fff1b8; border-bottom: 1.5px solid #d4a017; }
sup.cid { color: #b3261e; font-weight: bold; font-family: 'DejaVu Sans', sans-serif; font-size: 6.5pt; margin-left: 1px; }
sup.nid { color: #1f5fa8; font-weight: bold; font-family: 'DejaVu Sans', sans-serif; font-size: 6.5pt; }
.mnote { float: right; clear: right; width: 1.75in; margin-right: -1.95in; margin-top: 2px; margin-bottom: 4px;
  font-family: 'DejaVu Sans', sans-serif; font-size: 6.6pt; line-height: 1.28; text-align: left; font-style: normal; font-weight: normal;
  padding: 3px 5px; border-radius: 3px; color: #333; text-indent: 0; }
.corr-note { background: #fdf3f2; border-left: 3px solid #b3261e; } .corr-note b:first-child { color: #b3261e; }
.note-note { background: #eef4fb; border-left: 3px solid #1f5fa8; } .note-note b:first-child { color: #1f5fa8; }
.wide { padding-right: 0; margin-right: -1.95in; }
table.reg { font-size: 7pt; font-family: 'DejaVu Sans', sans-serif; width: 100%; } table.reg td { padding: 2px 4px; }
.title { text-align: center; padding-top: 0.9in; margin-right: -1.95in; }
.title h1 { border: none; font-size: 23pt; margin-bottom: 6px; }
.title .sub { font-family: 'DejaVu Sans', sans-serif; font-size: 12.5pt; color: #2b4f75; }
.title .authors { margin-top: 26px; font-family: 'DejaVu Sans', sans-serif; font-size: 15pt; font-weight: bold; }
.title .affil, .title .meta { font-size: 10pt; color: #444; margin-top: 6px; line-height: 1.5; }
.title .ed { margin-top: 26px; font-size: 13.5pt; color: #123a63; font-family: 'DejaVu Sans', sans-serif; font-weight: bold; }
.title .box { margin: 30px auto 0; width: 78%; font-size: 8.8pt; color: #444; border-top: 1px solid #b9c3cf; padding-top: 10px; text-align: justify; }
.credits, .toc { break-before: page; margin-right: -1.95in; }
.toc li { list-style: none; margin: 4px 0; font-size: 10.2pt; } .toc ul { padding-left: 0; } .toc ul ul { padding-left: 20px; }
.toc ul ul li { font-size: 9pt; margin: 1px 0; } .toc .pg { float: right; }
.loc { font-weight: normal; color: #777; font-size: 8.5pt; }
.k, .kn, .kd { color: #0b4ea2; font-weight: bold; } .c, .c1, .cm, .ch { color: #6a737d; font-style: italic; } .s, .s1, .s2, .sd, .sa { color: #a14b00; }
.mi, .mf, .m { color: #116329; } .nf, .nc { color: #6f42c1; } .nb, .bp { color: #005cc5; } .o, .ow { color: #a626a4; }
"""


def title_html():
    return """
<div class="title">
  <h1>A Review of Linear Mixed Models</h1>
  <div class="sub">Python Companion to Chapter 1 (draft, March 2025)</div>
  <div class="sub" style="font-size:10.5pt;margin-top:4px">Genetic Data Analysis for Plant and Animal Breeding, second edition (in preparation)</div>
  <div class="authors">Fikret Isik · James B. Holland · Christian Maltecca</div>
  <div class="affil">North Carolina State University</div>
  <div class="meta">Original chapter draft and data © the authors. First edition: Springer, 2017.</div>
  <div class="ed">Python Companion<br/><span style="font-weight:normal;font-size:10pt">statsmodels · pyreml · scikit-learn · NumPy/SciPy, with a validated custom package (lmmtools)</span></div>
  <div class="box">All concepts, examples, data and interpretations are the authors'. This unofficial companion re-implements the chapter's
  SAS, R and ASReml-R analyses in Python, checks every reproducible number against the draft, and marks corrections in the margin.
  It has not been reviewed or endorsed by the authors. The authors share the draft chapters for personal research use and invite
  feedback; the corrections register (Appendix D) is written so it can be passed on to them.</div>
</div>"""


def credits_html():
    return """
<div class="credits">
  <h1>About this companion</h1>
  <h3>Original work</h3>
  <p><b>Isik, F., Holland, J.B. and Maltecca, C.</b> Chapter 1, “A Review of Linear Mixed Models”, first draft (March 2025), for the second
  edition of <i>Genetic Data Analysis for Plant and Animal Breeding</i>. The draft chapters are distributed by the authors through
  F. Isik's NC State faculty web page for personal research use. First edition: Isik, Holland &amp; Maltecca (2017), Springer.
  The lactation example follows Robinson (1991); the maize RIL data are the authors'.</p>
  <h3>How to read this document</h3>
  <ul>
  <li><b>Narrative.</b> A condensed restatement of each section, in the chapter's order. Read it alongside the draft.</li>
  <li><b>Green boxes</b> list the Python modules and imports used in the section. <b>Grey boxes</b> show the chapter's original SAS, R or
      ASReml-R code. <b>Blue boxes</b> show the Python translation, followed by its actual output.</li>
  <li><b>Checks.</b> Each result printed in the chapter is compared automatically; <span class="pass">PASS</span> lines show the
      Python and chapter values, and any mismatch stops the notebook.</li>
  <li><b>Corrections</b> are <span class="corr">highlighted<sup class="cid">C#</sup></span> in the text with an explanation in the
      right margin (red). Software notes are marked <sup class="nid">N#</sup> (blue). All are collected in Appendix D.</li>
  <li><b>Custom code</b> is confined to the <code>lmmtools</code> package and is labelled CUSTOM wherever it is imported. Its design,
      unit tests and benchmarks are in Appendix C; alternatives for different use cases (small vs large data, structured
      covariances, GPU, Bayesian) are in Appendix B.</li>
  <li><b>Data.</b> Sources, verification and items to request from the authors: Appendix A.</li>
  </ul>
  <h3>Environment</h3>
  <p>Python 3.12 in a uv-managed virtual environment; numpy, scipy, pandas, statsmodels 0.15, scikit-learn 1.9, pyreml 0.5.5 (PyTorch
  backend), matplotlib. Every code listing in this document was executed to produce the output shown; the accompanying Jupyter notebooks
  contain the same code and annotations. Setup for Linux, macOS and Windows is in <code>README.md</code>.</p>
</div>"""


def h2_list(stem):
    nb = nbformat.read(NB / f"{stem}.ipynb", 4)
    return [l[3:].strip() for c in nb.cells if c.cell_type == "markdown" and not c.source.startswith("**Original")
            for l in c.source.splitlines() if l.startswith("## ")]


def toc_html(pages):
    items = []
    for title, stem, kind in PARTS:
        subs = "".join(f"<li>{html.escape(h)}<span class='pg'>{pages.get((title, h), '')}</span></li>" for h in (h2_list(stem) if stem else []))
        items.append(f"<li><b>{html.escape(title)}</b><span class='pg'>{pages.get(title, '')}</span><ul>{subs}</ul></li>")
    return f"<div class='toc'><h1>Contents</h1><ul>{''.join(items)}</ul></div>"


def nb_title(stem):
    if stem in FIRST_H1:
        return FIRST_H1[stem]
    nb = nbformat.read(NB / f"{stem}.ipynb", 4)
    return next(l[2:].strip() for c in nb.cells if c.cell_type == "markdown" for l in c.source.splitlines() if l.startswith("# "))


def resolve_titles():
    global PARTS
    PARTS = [(nb_title(stem) if stem else t, stem, k) for t, stem, k in PARTS]


def render(pages):
    body = title_html() + credits_html() + toc_html(pages)
    for title, stem, kind in PARTS:
        if kind in ("main", "app"):
            body += nb_html(stem, title)
        elif kind == "listing":
            body += listing_html()
        elif kind == "register":
            body += register_html()
        else:
            body += refs_html()
    doc = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    (ROOT / "build" / "companion.html").write_text(doc)
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page()
        pg.goto((ROOT / "build" / "companion.html").as_uri()); pg.wait_for_timeout(500)
        pg.pdf(path=str(OUT), format="Letter", print_background=True, display_header_footer=True, header_template="<div></div>",
               footer_template="<div style='width:100%;font-size:7pt;color:#777;text-align:center;font-family:sans-serif'>"
               "Python companion to Isik, Holland &amp; Maltecca, Ch. 1 “A Review of Linear Mixed Models” (draft 2025) · <span class='pageNumber'></span></div>",
               margin={"top": "0.7in", "bottom": "0.8in", "left": "0.7in", "right": "0.5in"})
        b.close()


def find_pages():
    n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(OUT)], capture_output=True, text=True).stdout).group(1))
    texts = [re.sub(r"\s+", " ", subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), str(OUT), "-"],
                                                 capture_output=True, text=True).stdout) for i in range(1, n + 1)]
    pages, last = {}, 4

    def locate(h, start):
        key = re.sub(r"\s+", " ", h)[:40]
        for i in range(start, n + 1):
            if key in texts[i - 1]:
                return i
    for title, stem, kind in PARTS:
        pg = locate(title, last)
        if pg: pages[title] = last = pg
        for h in (h2_list(stem) if stem else []):
            pg = locate(h, last)
            if pg: pages[(title, h)] = last = pg
    return pages, n


if __name__ == "__main__":
    resolve_titles()
    render({}); p1, _ = find_pages(); render(p1); p2, n = find_pages()
    print(f"{OUT.name}: {n} pages; TOC entries located {len(p2)}; stable {p1 == p2}")
