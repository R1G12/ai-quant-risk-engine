"""Plotly HTML overview for merged risk dataset (Phase 2 analytics)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def write_risk_dataset_overview(
    parquet_path: Path,
    output_html: Path,
    *,
    max_rows: int = 50_000,
) -> Path | None:
    """Write an interactive HTML report (returns coverage + feature null rates).

    Returns output path if Plotly is available, else None.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        LOGGER.warning("Plotly not installed; skipping analytics HTML export")
        return None

    lf = pl.scan_parquet(parquet_path)
    df = lf.head(max_rows).collect()
    if df.is_empty():
        LOGGER.warning("Risk dataset empty; skipping analytics export")
        return None

    numeric = [c for c, dt in df.schema.items() if dt.is_numeric() and c not in ("year", "month")]
    null_rates = (
        df.select([pl.col(c).null_count().alias(c) for c in numeric])
        .transpose(include_header=True, header_name="feature", column_names=["null_count"])
        .with_columns((pl.col("null_count") / df.height).alias("null_rate"))
        .sort("null_rate", descending=True)
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Null rate by feature", "Returns distribution (sample)"),
        vertical_spacing=0.12,
        row_heights=[0.45, 0.55],
    )
    fig.add_trace(
        go.Bar(
            x=null_rates["feature"].to_list(),
            y=null_rates["null_rate"].to_list(),
            name="null_rate",
        ),
        row=1,
        col=1,
    )
    if "returns" in df.columns:
        ret = df.filter(pl.col("returns").is_not_null())
        if not ret.is_empty():
            fig.add_trace(
                go.Histogram(x=ret["returns"].to_list(), name="returns", nbinsx=40),
                row=2,
                col=1,
            )

    fig.update_layout(
        title="Risk dataset feature overview (Phase 2)",
        template="plotly_dark",
        height=720,
        showlegend=False,
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html), include_plotlyjs="cdn")
    LOGGER.info("Analytics report written", extra={"path": str(output_html)})
    return output_html
