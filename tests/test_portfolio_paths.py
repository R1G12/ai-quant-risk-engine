"""Portfolio path helper tests."""

import numpy as np

from src.simulation.portfolio_paths.wealth import portfolio_paths_from_asset_paths, weights_from_dict


def test_weights_from_dict_order() -> None:
    w = weights_from_dict({"B": 0.3, "A": 0.7}, ["A", "B"])
    assert list(w) == [0.7, 0.3]


def test_portfolio_paths_einsum() -> None:
    asset = np.ones((2, 2, 3))
    asset[:, 0, :] = 2.0
    asset[:, 1, :] = 1.0
    weights = np.array([0.5, 0.5])
    port = portfolio_paths_from_asset_paths(asset, weights)
    assert port.shape == (2, 3)
    assert np.allclose(port, 1.5)
