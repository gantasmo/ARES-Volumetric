#!/usr/bin/env python3
"""
ARES Runtime Specification - build script.

Concatenates the numbered section files in this folder into a single
Markdown document and a self-contained, printable HTML document.

Usage:
    python build.py

Outputs (written to the parent folder):
    ../ARES-Runtime-Specification.md     canonical single-file Markdown
    ../ARES-Runtime-Specification.html   styled, printable (Save as PDF from any browser)

Mermaid diagrams:
    Fenced ```mermaid blocks render on GitHub / VS Code directly.
    The HTML build pulls mermaid from a CDN (needs network the first time you
    open it). For a fully offline PDF, install the Mermaid CLI and pandoc:
        npm i -g @mermaid-js/mermaid-cli
        pandoc ../ARES-Runtime-Specification.md -o out.pdf \
            --filter mermaid-filter --toc --number-sections
"""
import os
import glob
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
OUT_MD = os.path.join(PARENT, "ARES-Runtime-Specification.md")
OUT_HTML = os.path.join(PARENT, "ARES-Runtime-Specification.html")


def collect():
    files = sorted(glob.glob(os.path.join(HERE, "[0-9]*.md")))
    if not files:
        sys.exit("No section files (NN-*.md) found in spec/.")
    parts = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            parts.append(fh.read().rstrip() + "\n")
    return files, "\n\n".join(parts) + "\n"


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ARES Runtime Specification</title>
<style>
  :root {{ --fg:#1a1a1a; --muted:#555; --rule:#d0d0d0; --accent:#7c3a12;
           --code-bg:#f5f3f0; --th-bg:#efe9e2; }}
  html {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
  body {{ color:var(--fg); font: 15px/1.6 "Charter","Georgia",serif;
          max-width: 52em; margin: 0 auto; padding: 3em 2.5em; }}
  h1,h2,h3,h4 {{ font-family:"Segoe UI",Helvetica,Arial,sans-serif;
                 line-height:1.25; }}
  h1 {{ font-size:2.0em; border-bottom:3px solid var(--accent); padding-bottom:.2em; }}
  h2 {{ font-size:1.5em; margin-top:2em; border-bottom:1px solid var(--rule);
        padding-bottom:.15em; }}
  h3 {{ font-size:1.2em; margin-top:1.6em; }}
  h4 {{ font-size:1.02em; color:var(--muted); }}
  code {{ font-family:"Cascadia Code",Consolas,"Courier New",monospace;
          font-size:.88em; background:var(--code-bg); padding:.1em .35em;
          border-radius:3px; }}
  pre {{ background:var(--code-bg); padding:1em 1.2em; border-radius:6px;
         overflow-x:auto; border:1px solid var(--rule); }}
  pre code {{ background:none; padding:0; }}
  table {{ border-collapse:collapse; width:100%; margin:1.2em 0; font-size:.9em;
           display:block; overflow-x:auto; }}
  th,td {{ border:1px solid var(--rule); padding:.45em .7em; text-align:left;
           vertical-align:top; }}
  th {{ background:var(--th-bg); }}
  blockquote {{ border-left:4px solid var(--accent); margin:1em 0; padding:.2em 1em;
                color:var(--muted); background:#faf8f5; }}
  a {{ color:var(--accent); }}
  .mermaid {{ text-align:center; margin:1.4em 0; }}
  hr {{ border:none; border-top:1px solid var(--rule); margin:2em 0; }}
  @media print {{
    body {{ max-width:none; padding:0; font-size:10.5pt; }}
    h1,h2,h3,h4 {{ break-after:avoid; }}
    pre,table,blockquote {{ break-inside:avoid; }}
    h2 {{ break-before:page; }}
  }}
</style>
</head>
<body>
{body}
<script type="module">
  // Progressive enhancement: online, render Mermaid to inline SVG; offline, the
  // python-markdown code block stays put and remains perfectly readable.
  const blocks = [...document.querySelectorAll("pre > code.language-mermaid")];
  try {{
    const m = (await import("https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs")).default;
    m.initialize({{ startOnLoad: false, theme: "neutral" }});
    for (let i = 0; i < blocks.length; i++) {{
      try {{
        const {{ svg }} = await m.render("mmd" + i, blocks[i].textContent);
        const d = document.createElement("div");
        d.className = "mermaid";
        d.innerHTML = svg;
        blocks[i].closest("pre").replaceWith(d);
      }} catch (e) {{ /* leave this block as code */ }}
    }}
  }} catch (e) {{ /* offline: all mermaid blocks remain as styled code */ }}
</script>
</body>
</html>
"""


def to_html(md_text):
    try:
        import markdown
    except ImportError:
        sys.exit("pip install markdown  (needed for the HTML build)")
    html = markdown.markdown(
        md_text,
        extensions=["extra", "tables", "fenced_code", "toc", "sane_lists", "attr_list"],
        extension_configs={"toc": {"permalink": False}},
    )
    return HTML_TEMPLATE.format(body=html)


def main():
    files, md = collect()
    # The Markdown build must not depend on the HTML one: write it first, so a missing
    # `markdown` module still leaves a current .md (to_html exits the process).
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(md)
    # Render BEFORE opening the HTML: open(..., "w") truncates, so rendering inside the
    # `with` turned a missing `markdown` into a 0-byte ARES-Runtime-Specification.html.
    html = to_html(md)
    with open(OUT_HTML, "w", encoding="utf-8") as fh:
        fh.write(html)
    words = len(md.split())
    print(f"Combined {len(files)} sections.")
    print(f"  {OUT_MD}   ({words:,} words, ~{max(1, words // 500)} pages)")
    print(f"  {OUT_HTML}  (open in a browser, then Print -> Save as PDF)")


if __name__ == "__main__":
    main()
