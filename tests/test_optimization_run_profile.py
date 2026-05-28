"""Optimization stage respects run profile weighting."""

from __future__ import annotations

import numpy as np
import pytest

from src.risk.pipeline.optimization_stage import _max_sharpe_weights
from src.risk.optimization.constraints import PortfolioConstraints
from src.utils.config import load_app_config


def test_max_sharpe_partial_respects_anchors() -> None:
    app = load_app_config()
    if app.run is None or app.run.portfolio.weighting != "partial":
        pytest.skip("configs/run.yaml partial profile required")
    tickers = list(app.market.tickers)
    n = len(tickers)
    mean_r = np.full(n, 0.0005)
    cov = np.diag(np.full(n, 0.0004))
    cons = PortfolioConstraints(long_only=not app.run.portfolio.allow_shorts, max_weight=0.5)
    w, ok = _max_sharpe_weights(app, tickers, mean_r, cov, cons)
    assert ok
    for t, anchor in app.run.portfolio.anchor_weights.items():
        i = tickers.index(t)
        assert w[i] == pytest.approx(anchor, rel=0, abs=1e-4) or abs(w[i]) == pytest.approx(
            abs(anchor), rel=0, abs=1e-4
        )
