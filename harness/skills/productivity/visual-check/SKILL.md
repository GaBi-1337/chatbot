---
name: visual-check
description: "Look at plots, web pages, diagrams, PDFs before delivery."
version: 1.0.0
author: muse
platforms: [linux]
metadata:
  hermes:
    tags: [vision, plots, charts, screenshots, diagrams, verification]
    category: productivity
    related_skills: [web-app-testing, pdf, architecture-diagram]
---

# Visual check

A script that runs without errors can still draw an empty axis, overlapping labels, clipped text,
the wrong series or a blank page. Render the output to PNG, look at it with `vision_analyze`, fix,
and render again until the image shows what the data and the request say.

## Render to PNG

| Output | Command |
|---|---|
| matplotlib | `plt.savefig("out.png", dpi=100, bbox_inches="tight")`; scripts run headless, so `plt.show()` shows nothing |
| HTML file, SVG | `mcp__internet__web_screenshot(url="file:///abs/path.html", path="/tmp/shot.png")` |
| local web app | `mcp__internet__web_screenshot(url="localhost:PORT/page", ...)`; for clicks and flows load `web-app-testing` |
| phone layout | the same with `width=390, height=844` |
| PDF page N | `pdftoppm -f N -l N -r 80 -png -singlefile in.pdf /tmp/page` gives /tmp/page.png |
| Graphviz | `dot -Tpng graph.dot -o graph.png` |
| .docx, .pptx, .xlsx | `soffice --headless --convert-to pdf --outdir /tmp file.docx`, then the PDF row; a slide deck becomes one page per slide |

Keep images about 1300 px wide or smaller: every image you look at stays in the context. Look at
the pages or slides that changed, not all of them.

## Look

Ask `vision_analyze` for concrete findings, not approval. Name what should be there and ask what
is wrong:

- "List any text that overlaps, is cut off or is too small to read."
- "Read out the title, axis labels, units and legend entries."
- "How many bars or lines are there, and roughly what values does each reach?" Compare the answer
  with the numbers in the data.
- For pages: "Is anything misaligned, overflowing its box, or missing compared with: <spec>?"

When you deliver, say what you checked visually and what you could not render.
