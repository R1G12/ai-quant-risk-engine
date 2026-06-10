"""HMM regime detection (hmmlearn boundary)."""

from __future__ import annotations

import polars as pl

from src.utils.config import RiskConfig


def fit_hmm_regimes(
    returns: pl.Series,
    cfg: RiskConfig,
    seed: int = 42,
) -> pl.DataFrame:
    """Gaussian HMM on portfolio returns; labels ordered by mean return."""
    from hmmlearn.hmm import GaussianHMM

    r = returns.drop_nulls().to_numpy().reshape(-1, 1)
    if len(r) < cfg.hmm_n_regimes * 2:
        return pl.DataFrame({"timestamp": [], "regime": [], "regime_label": []})

    hmm = GaussianHMM(
        n_components=cfg.hmm_n_regimes,
        covariance_type="full",
        n_iter=max(cfg.hmm_n_iter, 300),
        tol=1e-3,
        random_state=seed,
        verbose=False,
    )
    hmm.fit(r)
    states = hmm.predict(r)
    ranking = sorted(range(cfg.hmm_n_regimes), key=lambda i: hmm.means_[i][0])
    labels_default = ["low", "mid", "high"]
    name_map = {
        ranking[i]: labels_default[i] if i < len(labels_default) else f"regime_{i}"
        for i in range(cfg.hmm_n_regimes)
    }

    labels = [name_map.get(s, str(s)) for s in states]
    return pl.DataFrame({"regime": states, "regime_label": labels})
