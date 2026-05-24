# Quantitative Methods

## Volatility

- **Rolling:** σ_t = std(r_{t-w:t}) × √252
- **EWMA:** σ²_t = λσ²_{t-1} + (1-λ)r²_t, λ = 1 - 2/(span+1)
- **Regime flag:** z-score of EWMA vol vs rolling median/std

## VaR / CVaR

- **Historical VaR:** quantile_{1-α}(r)
- **Parametric VaR:** μ + z_{1-α} σ
- **Monte Carlo VaR:** simulate N ~ N(μ, σ), empirical quantile
- **CVaR:** E[r | r ≤ VaR_α]

## Covariance shrinkage

Ledoit-Wolf style shrinkage toward scaled identity stabilizes Σ when T is small.

## Approved boundaries

- `numpy` / `scipy` for optimization and distributions
- `arch` for GARCH in `src/risk/volatility/garch.py`
- `hmmlearn` for regimes in `src/risk/regimes/hmm.py`
