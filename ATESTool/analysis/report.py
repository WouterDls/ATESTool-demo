"""One self-contained HTML run report bundling ``stats`` + plots.

:func:`write_html_report` needs no new dependency (stdlib only): every PNG is base64-embedded
directly into the page, so the resulting file has no external asset links and can be opened,
emailed, or archived entirely on its own, standalone in the literal sense.
"""

from __future__ import annotations

import base64
import html
import math
from pathlib import Path
from typing import Sequence

_CSS = """
body { font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 2rem; color: #1a1a1a; }
h1 { font-size: 1.4rem; }
h2 { font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid #ccc; padding-bottom: 0.25rem; }
table { border-collapse: collapse; margin: 0.5rem 0; }
td, th { border: 1px solid #ccc; padding: 0.3rem 0.8rem; text-align: left; }
th { background: #f2f2f2; }
pre { background: #f7f7f7; padding: 1rem; overflow-x: auto; white-space: pre-wrap; }
figure { margin: 1rem 0; }
figure img { max-width: 100%; border: 1px solid #ddd; }
figcaption { font-size: 0.85rem; color: #555; margin-top: 0.25rem; }
"""


def _stats_table(stats: dict[str, float]) -> str:
    rows = []
    for name, value in stats.items():
        display = "n/a" if isinstance(value, float) and math.isnan(value) else f"{value:g}"
        rows.append(f"<tr><td>{html.escape(str(name))}</td><td>{html.escape(display)}</td></tr>")
    return (
        "<table><tr><th>stat</th><th>value</th></tr>" + "".join(rows) + "</table>"
        if rows
        else "<p>(no stats)</p>"
    )


def _embedded_figure(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(path.stem)}">'
        f"<figcaption>{html.escape(path.name)}</figcaption></figure>"
    )


def write_html_report(
    path: Path | str,
    stats: dict[str, float],
    plot_paths: Sequence[Path],
    summary_text: str = "",
    title: str = "ATESTool run report",
) -> Path:
    """Write one self-contained HTML file: a stats table, an optional summary, then every plot.

    ``plot_paths`` are embedded in the given order (base64-encoded PNG bytes); no copying or
    relative-path bookkeeping needed, and the file stays readable after ``plot_paths``'
    originals are moved or deleted. Missing files are skipped with a placeholder note rather
    than raising, so one bad path does not sink the whole report.
    """
    path = Path(path)
    sections = [f"<h1>{html.escape(title)}</h1>"]

    sections.append("<h2>Stats</h2>")
    sections.append(_stats_table(stats))

    if summary_text:
        sections.append("<h2>Summary</h2>")
        sections.append(f"<pre>{html.escape(summary_text)}</pre>")

    sections.append(f"<h2>Plots ({len(plot_paths)})</h2>")
    for p in plot_paths:
        if Path(p).exists():
            sections.append(_embedded_figure(Path(p)))
        else:
            sections.append(f"<p><em>missing: {html.escape(str(p))}</em></p>")

    document = (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head>"
        f"<body>{''.join(sections)}</body></html>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path
