"""Single HTML research dashboard with chart picker and explanations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.analytics.charts.registry import build_chart_registry
from src.utils.config import AppConfig
from src.utils.logger import get_logger
from src.utils.paths import ensure_dir

LOGGER = get_logger(__name__)

_CSS = """
:root { --bg: #0f1419; --panel: #1a2332; --text: #e7ecf3; --muted: #9aa8bc; --accent: #3b82f6; }
* { box-sizing: border-box; }
body { margin: 0; font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); }
header { padding: 1rem 1.5rem; border-bottom: 1px solid #2a3544; }
header h1 { margin: 0 0 0.25rem; font-size: 1.35rem; }
header p { margin: 0; color: var(--muted); font-size: 0.9rem; }
.layout { display: grid; grid-template-columns: 260px 1fr 320px; min-height: calc(100vh - 72px); }
nav { background: var(--panel); padding: 1rem; overflow-y: auto; border-right: 1px solid #2a3544; }
nav h3 { font-size: 0.75rem; text-transform: uppercase; color: var(--muted); margin: 1rem 0 0.5rem; }
nav button { display: block; width: 100%; text-align: left; background: transparent; border: none;
  color: var(--text); padding: 0.5rem 0.65rem; border-radius: 6px; cursor: pointer; font-size: 0.9rem; }
nav button:hover, nav button.active { background: #243044; color: var(--accent); }
main { padding: 1rem; overflow: auto; }
aside { background: var(--panel); padding: 1rem 1.25rem; border-left: 1px solid #2a3544; overflow-y: auto; }
aside h2 { margin: 0 0 0.75rem; font-size: 1rem; }
aside p { color: var(--muted); line-height: 1.55; font-size: 0.9rem; }
#chart-root { min-height: 480px; }
a { color: var(--accent); }
"""


def write_unified_dashboard(
    output_dir: Path,
    figures: dict[str, object],
    registry: list,
    app: AppConfig,
) -> Path:
    """Write index.html embedding Plotly figures as JSON."""
    ensure_dir(output_dir)
    specs = {s.id: {"title": s.title, "description": s.description, "group": s.group} for s in registry}
    fig_json: dict[str, str] = {}
    for cid, fig in figures.items():
        fig_json[cid] = fig.to_json()

    groups: dict[str, list] = {}
    for s in registry:
        if s.id in fig_json:
            groups.setdefault(s.group, []).append(s.id)

    nav_html = []
    for group, ids in groups.items():
        nav_html.append(f"<h3>{group}</h3>")
        for cid in ids:
            title = specs[cid]["title"]
            nav_html.append(f'<button type="button" data-chart="{cid}">{title}</button>')

    index_path = output_dir / "index.html"
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Research Dashboard — {app.research.meta.experiment_id}</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>{_CSS}</style>
</head>
<body>
  <header>
    <h1>Quant Research Dashboard</h1>
    <p>Experiment: <strong>{app.research.meta.experiment_id}</strong> · Generated {generated} ·
    <a href="../../research/reports/REPORT.md">Read REPORT.md</a></p>
  </header>
  <div class="layout">
    <nav id="nav">{''.join(nav_html)}</nav>
    <main><div id="chart-root"></div></main>
    <aside>
      <h2 id="chart-title">Select a chart</h2>
      <p id="chart-desc">Use the sidebar to explore risk, simulation, backtest, and stress views.</p>
    </aside>
  </div>
  <script>
    const specs = {json.dumps(specs)};
    const figures = {json.dumps(fig_json)};
    const chartIds = {json.dumps([s.id for s in registry if s.id in fig_json])};
    function showChart(id) {{
      if (!figures[id]) return;
      document.querySelectorAll('nav button').forEach(b => b.classList.toggle('active', b.dataset.chart === id));
      Plotly.react('chart-root', JSON.parse(figures[id]).data, JSON.parse(figures[id]).layout);
      document.getElementById('chart-title').textContent = specs[id].title;
      document.getElementById('chart-desc').textContent = specs[id].description;
    }}
    document.getElementById('nav').addEventListener('click', e => {{
      if (e.target.dataset.chart) showChart(e.target.dataset.chart);
    }});
    if (chartIds.length) showChart(chartIds[0]);
  </script>
</body>
</html>"""
    index_path.write_text(html, encoding="utf-8")
    LOGGER.info("Unified dashboard written", extra={"path": str(index_path)})
    return index_path
