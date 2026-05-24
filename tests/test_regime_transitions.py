"""Regime transition matrix tests."""

import numpy as np

from src.analytics.charts.registry import _build_regime_transition
from src.utils.config import load_app_config


def test_regime_transition_matrix_rows_sum_to_one() -> None:
    app = load_app_config()
    fig = _build_regime_transition(app)
    if fig is None:
        return
    z = fig.data[0].z
    row_sums = np.array(z).sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)
