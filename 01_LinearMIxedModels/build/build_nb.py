"""Build annotated notebooks from build/src/*.txt and execute them.
Markers: %%md | %%imports | %%code [nopdf] | %%orig <Lang> [| title]
Inline corrections in markdown: {{C6|corrected text}} ; clarification notes: {{N1}}"""
import re, sys, time, pathlib
import nbformat
from nbclient import NotebookClient
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from corrections import CORRECTIONS, NOTES

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "build" / "src", ROOT / "notebooks"


def parse(path):
    cells, kind, buf = [], None, []
    for line in path.read_text().splitlines():
        if line.startswith("%%"):
            if kind: cells.append((kind, "\n".join(buf).strip("\n")))
            kind, buf = line[2:].strip(), []
        else:
            buf.append(line)
    if kind: cells.append((kind, "\n".join(buf).strip("\n")))
    return cells


def md_for_notebook(text):
    used = []
    def corr(m):
        cid, txt = m.group(1), m.group(2)
        used.append(cid)
        return f'<mark>{txt}</mark><sup>[{cid}]</sup>'
    def note(m):
        used.append(m.group(1)); return f"<sup>[{m.group(1)}]</sup>"
    text = re.sub(r"\{\{(C\d+)\|(.+?)\}\}", corr, text, flags=re.S)
    text = re.sub(r"\{\{(N\d+)\}\}", note, text)
    if used:
        lines = []
        for cid in dict.fromkeys(used):
            if cid.startswith("C"):
                c = CORRECTIONS[cid]
                lines.append(f"> **Correction {cid}** ({c['where']}). Draft: {c['original']}. Corrected: {c['corrected']}. {c['why']}")
            else:
                n = NOTES[cid]; lines.append(f"> **Note {cid}** ({n['where']}). {n['text']}")
        text += "\n\n" + "\n>\n".join(lines)
    return text


def to_notebook(cells):
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "lmm-companion", "display_name": "LMM companion (uv)", "language": "python"}
    for kind, text in cells:
        if kind == "md":
            nb.cells.append(nbformat.v4.new_markdown_cell(md_for_notebook(text)))
        elif kind.startswith("orig"):
            lang, _, title = (x.strip() for x in kind[4:].partition("|"))
            lang = lang or "SAS"
            head = f"**Original {lang} code (from the chapter)" + (f" — {title}" if title else "") + "**"
            fence = {"SAS": "sas", "R": "r"}.get(lang, "text")
            c = nbformat.v4.new_markdown_cell(f"{head}\n\n```{fence}\n{text}\n```")
            c.metadata["orig"] = {"lang": lang, "title": title}
            nb.cells.append(c)
        elif kind.startswith("imports") or kind.startswith("code"):
            c = nbformat.v4.new_code_cell(text)
            tags = []
            if kind.startswith("imports"): tags.append("imports")
            if "nopdf" in kind: tags.append("nopdf")
            if tags: c.metadata["tags"] = tags
            nb.cells.append(c)
    return nb


def main(names):
    OUT.mkdir(exist_ok=True)
    srcs = sorted(SRC.glob("*.txt"))
    if names: srcs = [s for s in srcs if any(s.stem.startswith(n) for n in names)]
    for s in srcs:
        nb = to_notebook(parse(s)); t = time.time()
        NotebookClient(nb, timeout=3600, kernel_name="lmm-companion", resources={"metadata": {"path": str(ROOT)}}).execute()
        nbformat.write(nb, OUT / f"{s.stem}.ipynb")
        txt = "".join(o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []))
        print(f"{s.stem}: executed in {time.time()-t:.0f}s, {txt.count('PASS ')} checks passed, {txt.count('FAIL ')} failed", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
