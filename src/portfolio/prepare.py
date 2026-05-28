"""Materialize run profile into holdings and manifest artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from src.market.ticker_validation import format_skip_messages, require_min_tickers, validate_market_tickers
from src.portfolio.weights import exposure_summary, resolve_weights
from src.utils.config import AppConfig, market_source_for_run_mode
from src.utils.logger import get_logger
from src.utils.paths import HOLDINGS_PATH, PROJECT_ROOT, ensure_dir

LOGGER = get_logger(__name__)


def _holdings_path(app: AppConfig) -> Path:
    path = Path(app.risk.portfolio.holdings_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _config_fingerprint(app: AppConfig) -> str:
    run = app.run
    if run is None:
        return ""
    payload = {
        "mode": run.mode,
        "tickers": app.market.tickers,
        "weighting": run.portfolio.weighting,
        "anchor_weights": run.portfolio.anchor_weights,
        "manual_weights": run.portfolio.manual_weights,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def materialize_run(app: AppConfig) -> Path:
    """Write holdings.parquet and run_manifest.json from run profile."""
    if app.run is None:
        raise ValueError("No run profile loaded; use configs/run.yaml or --profile")

    requested = list(app.market.tickers)
    if not requested:
        raise ValueError("market.tickers is empty")

    validation_source = market_source_for_run_mode(app.run.mode)
    if app.market.source != validation_source:
        LOGGER.warning(
            "Validating tickers for run mode %r (%s), not shell market.source=%r",
            app.run.mode,
            validation_source,
            app.market.source,
        )
    filter_result = validate_market_tickers(requested, validation_source)
    for line in format_skip_messages(filter_result):
        LOGGER.warning(line)
    require_min_tickers(filter_result, context="aqre prepare")
    tickers = filter_result.valid

    port = app.run.portfolio
    weighting = port.weighting
    anchor_weights = {k: v for k, v in (port.anchor_weights or {}).items() if k in tickers}
    manual_weights = {k: v for k, v in (port.manual_weights or {}).items() if k in tickers}
    position_sides = {k: v for k, v in (port.position_sides or {}).items() if k in tickers}
    dropped_anchors = set(port.anchor_weights or {}) - set(anchor_weights)
    if dropped_anchors:
        LOGGER.warning("Dropped anchor_weights for skipped tickers: %s", sorted(dropped_anchors))

    kw = dict(
        allow_shorts=port.allow_shorts,
        max_gross_per_ticker=port.max_gross_per_ticker,
        position_sides=position_sides or None,
        manual_weights=manual_weights or None,
        anchor_weights=anchor_weights or None,
        risk_free=port.risk_free,
    )

    if weighting in ("equal", "manual"):
        weights = resolve_weights(weighting, tickers, **kw)
    else:
        # Placeholder for early pipeline stages; optimization stage applies full logic.
        weights = resolve_weights("equal", tickers, **kw)

    holdings_path = _holdings_path(app)
    ensure_dir(holdings_path.parent)
    pl.DataFrame({"asset": tickers, "weight": weights.tolist()}).write_parquet(holdings_path)

    manifest_path = PROJECT_ROOT / "data" / "run_manifest.json"
    ensure_dir(manifest_path.parent)
    exp = exposure_summary(weights)
    manifest = {
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": _config_fingerprint(app),
        "mode": app.run.mode,
        "market_source": validation_source,
        "tickers_requested": requested,
        "tickers": tickers,
        "skipped_tickers": filter_result.skipped,
        "skip_reasons": filter_result.reasons,
        "weighting": weighting,
        "holdings_path": str(holdings_path.relative_to(PROJECT_ROOT)),
        "weights": {t: float(w) for t, w in zip(tickers, weights, strict=False)},
        "exposure": exp,
        "note": (
            "partial/optimised use placeholder equal weights until optimize_portfolios runs"
            if weighting in ("partial", "optimised")
            else None
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return holdings_path
