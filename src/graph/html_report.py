"""Write a single self-contained HTML brief with narrative, charts, and cited sources."""

from __future__ import annotations

import html as html_mod
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import markdown

from src.utils.config import LeadershipAgentConfig


def _extract_source_lines(merged_context: str) -> List[str]:
    """Unique [SOURCE n: …] and [WEB SOURCE n: …] lines from tool output."""
    seen: set[str] = set()
    out: list[str] = []
    for line in (merged_context or "").splitlines():
        line = line.strip()
        if re.match(r"^\[SOURCE \d+:", line) or re.match(r"^\[WEB SOURCE \d+:", line):
            if line not in seen:
                seen.add(line)
                out.append(line)
    return out


def write_leadership_html_report(
    *,
    question: str,
    final_answer: str,
    merged_context: str,
    analysis_stdout: str,
    analysis_stderr: str,
    image_paths: List[str],
    cfg: LeadershipAgentConfig,
    project_root: Path,
) -> Path:
    reports_dir = cfg.resolve(cfg.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    html_path = reports_dir / f"leadership_report_{ts}.html"

    body_md = (final_answer or "").strip()
    body_html = markdown.markdown(body_md)

    figures_html: list[str] = []
    for i, abs_p in enumerate(image_paths):
        try:
            rel = Path(abs_p).resolve().relative_to(html_path.parent.resolve())
            src = str(rel).replace("\\", "/")
        except ValueError:
            src = Path(abs_p).name
        figures_html.append(
            f'<figure class="chart"><img src="{html_mod.escape(src)}" alt="Chart {i+1}"/>'
            f"<figcaption>Figure {i+1}</figcaption></figure>"
        )

    sources = _extract_source_lines(merged_context)
    sources_html = ""
    if sources:
        items = "".join(f"<li>{html_mod.escape(s)}</li>" for s in sources)
        sources_html = f"<section class='sources'><h2>Sources (from retrieval)</h2><ul>{items}</ul></section>"

    aux = ""
    if (analysis_stdout or "").strip():
        aux += f"<pre class='viz-stdout'>{html_mod.escape(analysis_stdout.strip()[:4000])}</pre>"
    if (analysis_stderr or "").strip():
        aux += f"<pre class='viz-stderr'>{html_mod.escape(analysis_stderr.strip()[:2000])}</pre>"

    title = html_mod.escape("Leadership brief — Adobe")
    q = html_mod.escape(question or "")

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, Segoe UI, Roboto, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
    h1 {{ font-size: 1.5rem; border-bottom: 1px solid #ccc; padding-bottom: 0.5rem; }}
    .question {{ background: #f5f5f5; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; }}
    .answer {{ line-height: 1.55; }}
    .charts {{ margin: 2rem 0; }}
    .chart img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 6px; }}
    .chart figcaption {{ font-size: 0.85rem; color: #666; margin-top: 0.35rem; }}
    .sources ul {{ padding-left: 1.2rem; }}
    .sources li {{ margin: 0.35rem 0; font-size: 0.95rem; }}
    pre {{ font-size: 0.8rem; overflow: auto; background: #fafafa; padding: 0.75rem; border-radius: 6px; }}
    .viz-stderr {{ color: #7a2e2e; }}
  </style>
</head>
<body>
  <h1>Leadership brief</h1>
  <p class="meta"><time>{html_mod.escape(ts)}</time> UTC</p>
  <section class="question"><strong>Question</strong><p>{q}</p></section>
  <section class="answer">{body_html}</section>
  <section class="charts"><h2>Charts &amp; analysis</h2>{"".join(figures_html) if figures_html else "<p><em>No figures produced for this run.</em></p>"}</section>
  {sources_html}
  {aux}
</body>
</html>"""
    html_path.write_text(page, encoding="utf-8")
    return html_path
