"""One-off patcher for trading_risk_manager_final.ipynb (plan implementation)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "trading_risk_manager_final.ipynb"


def src_join(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_src(cell: dict, text: str) -> None:
    cell["source"] = [line + "\n" for line in text.splitlines()]
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip("\n") + "\n"


PHASE2A_CELL = '''prices = download_close_prices(TICKERS, START, END).dropna(how='all')
missing = [t for t in TICKERS if t not in prices.columns or prices[t].dropna().empty]
if missing:
    print(f'⚠️  No price data for: {missing}')
prices = prices[[t for t in TICKERS if t in prices.columns]].dropna()
rets   = prices.pct_change().dropna()
print(f'Downloaded {len(prices)} days for {prices.shape[1]}/{len(TICKERS)} tickers.')
if prices.empty:
    raise ValueError(
        'No market data downloaded. Install: pip install -e ".[market]" '
        'or set YFINANCE_SSL_VERIFY=0 for dev (see src/market/adapters/yfinance.py).'
    )

mu_daily = rets.mean()
cov_daily = rets.cov()

_kw = dict(
    allow_shorts=ALLOW_SHORTS,
    max_gross_per_ticker=MAX_GROSS_PER_TICKER,
    position_sides=POSITION_SIDES,
    manual_weights=MANUAL_WEIGHTS,
    anchor_weights=ANCHOR_WEIGHTS,
    risk_free=RISK_FREE,
)

try:
    if PORTFOLIO_WEIGHTING == 'optimised':
        weights = resolve_weights('optimised', TICKERS, w_optimised=w_sharpe, **_kw)
    elif PORTFOLIO_WEIGHTING == 'partial':
        try:
            weights = resolve_weights('partial', TICKERS, w_optimised=w_sharpe, **_kw)
        except NameError:
            print('⚠️  w_sharpe not yet computed — optimizing partial weights inline.')
            weights, _ = optimize_partial_weights(
                mu_daily.values * 252, rets.cov().values * 252, TICKERS, ANCHOR_WEIGHTS, **_kw
            )
    else:
        weights = resolve_weights(PORTFOLIO_WEIGHTING, TICKERS, **_kw)
except NameError:
    print('⚠️  w_sharpe not yet computed — falling back to equal gross weights.')
    print('   Run Phase 3 first for optimised/partial, then re-run this cell.')
    weights = resolve_weights('equal', TICKERS, **_kw)

weights = np.asarray(weights, dtype=float)
exp = exposure_summary(weights)
portfolio_rets = (rets * weights).sum(axis=1)

print(f'\\n📐 Portfolio weights ({PORTFOLIO_WEIGHTING}):')
for t, w in zip(TICKERS, weights):
    side = 'short' if w < 0 else 'long'
    print(f'  {t}: {w:+.1%}  ({side})')
print(f'  Gross: {exp["gross"]:.2%}  |  Net: {exp["net"]:+.2%}  |  Long: {exp["long"]:+.2%}  |  Short: {exp["short"]:+.2%}')
print(f'\\nPortfolio return series: {len(portfolio_rets)} days')
print(f'  Annual mean : {portfolio_rets.mean()*252:.2%}')
print(f'  Annual vol  : {portfolio_rets.std()*np.sqrt(252):.2%}')
'''


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    # Force cell 8 = phase 2a
    if len(nb["cells"]) > 8 and nb["cells"][8]["cell_type"] == "code":
        set_src(nb["cells"][8], PHASE2A_CELL)

    for c in nb["cells"]:
        s = src_join(c)

        # Config cell
        if c["cell_type"] == "code" and "TICKERS   = " in s and "download_close_prices" in s:
            if "ALLOW_SHORTS" not in s:
                extra = """
# Gross budget: sum(|weight|) = 1. Negative weight = short (requires ALLOW_SHORTS).
ALLOW_SHORTS = True
MAX_GROSS_PER_TICKER = 0.50
POSITION_SIDES = {}  # e.g. {'QQQ': 'short'} — force sign after optimization

# Partial weights: fix some tickers, optimize the rest (mode='partial')
ANCHOR_WEIGHTS = {
    'CVX': 0.35,
    'GLD': 0.20,
}
SHOW_PER_TICKER_MC = False
"""
                s = s.replace("# ── Portfolio weighting", extra + "\n# ── Portfolio weighting")
                s = s.replace(
                    "# 'optimised' → use max-Sharpe weights from Phase 3",
                    "# 'partial'   → fix ANCHOR_WEIGHTS, optimize the rest (run Phase 3 first)\n"
                    "# 'optimised' → use max-Sharpe weights from Phase 3",
                )
                s = s.replace(
                    "from src.utils.config import MarketConfig\n",
                    "from src.utils.config import MarketConfig\n\n"
                    "from notebooks.trading_notebook_utils import (\n"
                    "    aggregate_portfolio_paths,\n"
                    "    exposure_summary,\n"
                    "    gbm_paths,\n"
                    "    optimize_max_sharpe_gross,\n"
                    "    optimize_partial_weights,\n"
                    "    resolve_weights,\n"
                    "    side_for_ticker,\n"
                    "    simulate_trailing_stops,\n"
                    ")\n",
                )
                set_src(c, s)

        # Phase 1 sentiment
        if c["cell_type"] == "code" and "get_ticker_headlines" in s and "portfolio_sentiment" in s:
            old = """STOPS, stop_tag = build_stops(portfolio_sentiment)
print(f'\\n🎯 Trailing stops ({stop_tag}):')
print(f'  {"Level":>10}   {"Exit fraction":>14}   Label')
print('  ' + '─'*42)
for s in STOPS:
    print(f'  {s["level"]:>+10.1%}   {s["exit_fraction"]:>14.1%}   {s["label"]}')"""
            new = """stops_by_ticker = {}
tags_by_ticker = {}
for t in TICKERS:
    score = float(ticker_sentiment.get(t, 0.0))
    stops_by_ticker[t], tags_by_ticker[t] = build_stops(score)[:2]

# Portfolio-level summary (optional)
STOPS, stop_tag = build_stops(portfolio_sentiment)

print('\\n🎯 Per-ticker trailing stops (sentiment-derived):')
print(f'  {"Ticker":<6} {"Sent":>7}  {"Regime":<18}  Stop levels')
print('  ' + '─' * 55)
for t in TICKERS:
    lvls = ', '.join(f'{s["level"]:+.0%}' for s in stops_by_ticker[t])
    print(f'  {t:<6} {ticker_sentiment.get(t, 0):>+7.3f}  {tags_by_ticker[t]:<18}  {lvls}')"""
            if old in s:
                s = s.replace(old, new)
                set_src(c, s)

        # Phase 2a (skip if already using utils resolve_weights)
        if False and c["cell_type"] == "code" and "download_close_prices" in s and "resolve_weights" in s:
            new_cell = '''prices = download_close_prices(TICKERS, START, END).dropna(how='all')
missing = [t for t in TICKERS if t not in prices.columns or prices[t].dropna().empty]
if missing:
    print(f'⚠️  No price data for: {missing}')
prices = prices[[t for t in TICKERS if t in prices.columns]].dropna()
rets   = prices.pct_change().dropna()
print(f'Downloaded {len(prices)} days for {prices.shape[1]}/{len(TICKERS)} tickers.')
if prices.empty:
    raise ValueError(
        'No market data downloaded. Install: pip install -e ".[market]" '
        'or set YFINANCE_SSL_VERIFY=0 for dev (see src/market/adapters/yfinance.py).'
    )

mu_daily = rets.mean()
cov_daily = rets.cov()

def _try_resolve():
    kw = dict(
        allow_shorts=ALLOW_SHORTS,
        max_gross_per_ticker=MAX_GROSS_PER_TICKER,
        position_sides=POSITION_SIDES,
        manual_weights=MANUAL_WEIGHTS,
        anchor_weights=ANCHOR_WEIGHTS,
        risk_free=RISK_FREE,
    )
    if PORTFOLIO_WEIGHTING == 'optimised':
        return resolve_weights('optimised', TICKERS, w_optimised=w_sharpe, **kw)
    if PORTFOLIO_WEIGHTING == 'partial':
        try:
            return resolve_weights('partial', TICKERS, w_optimised=w_sharpe, **kw)
        except NameError:
            print('⚠️  w_sharpe not yet computed — optimizing partial weights inline.')
            w, _ = optimize_partial_weights(
                mu_daily.values * 252,
                rets.cov().values * 252,
                TICKERS,
                ANCHOR_WEIGHTS,
                risk_free=RISK_FREE,
                allow_shorts=ALLOW_SHORTS,
                max_gross_per_ticker=MAX_GROSS_PER_TICKER,
                position_sides=POSITION_SIDES,
            )
            return w
    return resolve_weights(PORTFOLIO_WEIGHTING, TICKERS, **kw)

try:
    weights = _try_resolve()
except NameError:
    print('⚠️  w_sharpe not yet computed — falling back to equal gross weights.')
    print('   Run Phase 3 first for optimised/partial, then re-run this cell.')
    weights = resolve_weights('equal', TICKERS, allow_shorts=ALLOW_SHORTS,
        max_gross_per_ticker=MAX_GROSS_PER_TICKER, position_sides=POSITION_SIDES)

weights = np.asarray(weights, dtype=float)
exp = exposure_summary(weights)
portfolio_rets = (rets * weights).sum(axis=1)

print(f'\\n📐 Portfolio weights ({PORTFOLIO_WEIGHTING}):')
for t, w in zip(TICKERS, weights):
    side = side_for_ticker(t, POSITION_SIDES) if w == 0 else ('short' if w < 0 else 'long')
    print(f'  {t}: {w:+.1%}  ({side})')
print(f'  Gross: {exp["gross"]:.2%}  |  Net: {exp["net"]:+.2%}  |  Long: {exp["long"]:+.2%}  |  Short: {exp["short"]:+.2%}')
print(f'\\nPortfolio return series: {len(portfolio_rets)} days')
print(f'  Annual mean : {portfolio_rets.mean()*252:.2%}')
print(f'  Annual vol  : {portfolio_rets.std()*np.sqrt(252):.2%}')
'''
            set_src(c, new_cell)

        # Phase 3
        if c["cell_type"] == "code" and "from scipy.optimize import minimize" in s and "w_sharpe = opt.x" in s:
            new_cell = '''mu_annual  = rets.mean() * 252
cov_annual = rets.cov()  * 252
n          = len(TICKERS)

for t in TICKERS:
    if t in ticker_sentiment.index:
        mu_annual[t] += ticker_sentiment[t] * SENTIMENT_RETURN_BOOST

def portfolio_stats(w):
    ret    = w @ mu_annual
    vol    = np.sqrt(w @ cov_annual @ w)
    sharpe = (ret - RISK_FREE) / vol if vol > 1e-12 else 0.0
    return ret, vol, sharpe

if PORTFOLIO_WEIGHTING == 'partial':
    w_sharpe, ok = optimize_partial_weights(
        mu_annual.values, cov_annual.values, TICKERS, ANCHOR_WEIGHTS,
        risk_free=RISK_FREE, allow_shorts=ALLOW_SHORTS,
        max_gross_per_ticker=MAX_GROSS_PER_TICKER, position_sides=POSITION_SIDES,
    )
else:
    w_sharpe, ok = optimize_max_sharpe_gross(
        mu_annual.values, cov_annual.values,
        risk_free=RISK_FREE, allow_shorts=ALLOW_SHORTS,
        max_gross_per_ticker=MAX_GROSS_PER_TICKER,
    )

# Random portfolios on gross simplex
sim_w = np.random.dirichlet(np.ones(n), N_SIM_EF)
if ALLOW_SHORTS:
  signs = np.array([1.0 if POSITION_SIDES.get(t, 'long') == 'long' else -1.0 for t in TICKERS])
  sim_w = sim_w * signs
sim_stats = np.array([portfolio_stats(w) for w in sim_w])

fig = go.Figure()
fig.add_trace(go.Scatter(x=sim_stats[:,1], y=sim_stats[:,0], mode='markers',
    marker=dict(color=sim_stats[:,2], colorscale='Viridis', size=4,
                showscale=True, colorbar=dict(title='Sharpe')),
    name=f'{N_SIM_EF} random portfolios'))
r_opt, v_opt, s_opt = portfolio_stats(w_sharpe)
fig.add_trace(go.Scatter(x=[v_opt], y=[r_opt], mode='markers+text',
    marker=dict(color='red', size=14, symbol='star'),
    text=['Max Sharpe'], textposition='top right', name='Max Sharpe'))
fig.update_layout(title='Efficient Frontier (gross budget, sentiment-enhanced)',
                  xaxis_title='Volatility', yaxis_title='Return',
                  xaxis_tickformat='.0%', yaxis_tickformat='.0%',
                  template='plotly_dark')
fig.show()

exp = exposure_summary(w_sharpe)
print('\\n📌 Optimised weights:')
for t, w in zip(TICKERS, w_sharpe):
    print(f'  {t}: {w:+.1%}')
print(f'  Return: {r_opt:.2%}  |  Vol: {v_opt:.2%}  |  Sharpe: {s_opt:.2f}')
print(f'  Gross: {exp["gross"]:.2%}  |  Net: {exp["net"]:+.2%}')
print('\\n💡 Re-run Phase 2a to apply optimised/partial weights.')
'''
            set_src(c, new_cell)

        # Phase 4 MC
        if c["cell_type"] == "code" and "paths_stopped" in s and "stops_sorted = sorted(STOPS" in s:
            new_cell = '''if RANDOM_SEED is not None:
    np.random.seed(RANDOM_SEED)

path_by_ticker = {}
for ti, t in enumerate(TICKERS):
    side = 'short' if weights[ti] < 0 else 'long'
    ann_mu = float(rets[t].mean() * 252)
    ann_vol = float(rets[t].std() * np.sqrt(252))
    rp = REGIME_PARAMS.get(current_regime, REGIME_PARAMS.get('Neutral ⚪', list(REGIME_PARAMS.values())[1]))
    drift = rp['drift'] / 252 + float(ticker_sentiment.get(t, 0)) * SENTIMENT_DRIFT_NUDGE
    vol_d = max(ann_vol, 1e-6) / np.sqrt(252)
    alloc = abs(weights[ti]) * PORT_VAL
    raw = gbm_paths(
        horizon=HORIZON, n_paths=N_PATHS, start_value=alloc,
        drift_daily=drift, vol_daily=vol_d,
        seed=None if RANDOM_SEED is None else RANDOM_SEED + ti,
        side=side,
    )
    stops_t = stops_by_ticker.get(t, STOPS)
    path_by_ticker[t] = simulate_trailing_stops(raw, stops_t, side='long')

paths_stopped = aggregate_portfolio_paths(path_by_ticker, weights, TICKERS)
paths_raw = paths_stopped.copy()  # aggregate of stopped legs

final_stopped = paths_stopped[-1]
var95_stopped = np.percentile(final_stopped, 5)
days_ax = np.arange(HORIZON + 1)
pct = lambda arr, q: np.percentile(arr, q, axis=1)

# Use first ticker's stop ladder for legend labels
stops_sorted = sorted(stops_by_ticker.get(TICKERS[0], STOPS), key=lambda s: s['level'])
last_stop = np.zeros(N_PATHS, dtype=int)
for i in range(len(stops_sorted)):
    pass  # per-path stop coloring simplified below

STOP_COLOURS = {0: 'steelblue', 1: '#f0e68c', 2: '#ffa500', 3: '#ff4444'}
STOP_NAMES = {0: 'Portfolio path'}

fig = go.Figure()
idx = np.random.choice(N_PATHS, min(PLOT_SAMPLE_PATHS, N_PATHS), replace=False)
for i in idx:
    fig.add_trace(go.Scatter(x=days_ax, y=paths_stopped[:, i],
        line=dict(width=0.5, color='steelblue'), name='With stops',
        legendgroup='s', showlegend=(i == idx[0]), opacity=0.5))

for q, col, dash, lbl in [(95, 'lime', 'dot', '95th (w/ stops)'), (50, 'white', 'solid', 'Median'), (5, 'red', 'dot', '5th (VaR proxy)')]:
    fig.add_trace(go.Scatter(x=days_ax, y=pct(paths_stopped, q), mode='lines',
        line=dict(color=col, width=2, dash=dash), name=lbl))

fig.update_layout(
    title=(f'Monte Carlo — {N_PATHS:,} paths | Per-ticker stops | Regime: {current_regime}'),
    xaxis_title='Trading days', yaxis_title='Portfolio value ($)',
    yaxis_tickformat='$,.0f', template='plotly_dark')
fig.show()

if SHOW_PER_TICKER_MC:
    fig2 = go.Figure()
    for t in TICKERS:
        fig2.add_trace(go.Scatter(x=days_ax, y=path_by_ticker[t][:, 0], name=t, line=dict(width=1)))
    fig2.update_layout(title='Sample per-ticker stopped paths (path 0)', template='plotly_dark')
    fig2.show()

print(f'\\n📉 1-Year Risk Summary | Weights: {PORTFOLIO_WEIGHTING} | Regime: {current_regime}')
print(f'  VaR 95% (stopped aggregate): ${var95_stopped:,.0f}')
'''
            set_src(c, new_cell)

    # Force phase 3 (cell 20) and MC (cell 22)
    if len(nb["cells"]) > 20:
        s20 = src_join(nb["cells"][20])
        if "optimize_max_sharpe_gross" not in s20:
            for c in nb["cells"]:
                s = src_join(c)
                if c["cell_type"] == "code" and "from scipy.optimize import minimize" in s and "w_sharpe" in s:
                    set_src(nb["cells"][20], s.replace("from scipy.optimize import minimize\n", "").split("print('\\n💡")[0] + "print('\\n💡 Re-run Phase 2a to apply optimised/partial weights.')\n")
                    break
            phase3 = '''mu_annual  = rets.mean() * 252
cov_annual = rets.cov()  * 252
n          = len(TICKERS)

for t in TICKERS:
    if t in ticker_sentiment.index:
        mu_annual[t] += ticker_sentiment[t] * SENTIMENT_RETURN_BOOST

def portfolio_stats(w):
    ret    = w @ mu_annual
    vol    = np.sqrt(w @ cov_annual @ w)
    sharpe = (ret - RISK_FREE) / vol if vol > 1e-12 else 0.0
    return ret, vol, sharpe

if PORTFOLIO_WEIGHTING == 'partial':
    w_sharpe, ok = optimize_partial_weights(
        mu_annual.values, cov_annual.values, TICKERS, ANCHOR_WEIGHTS,
        risk_free=RISK_FREE, allow_shorts=ALLOW_SHORTS,
        max_gross_per_ticker=MAX_GROSS_PER_TICKER, position_sides=POSITION_SIDES,
    )
else:
    w_sharpe, ok = optimize_max_sharpe_gross(
        mu_annual.values, cov_annual.values,
        risk_free=RISK_FREE, allow_shorts=ALLOW_SHORTS,
        max_gross_per_ticker=MAX_GROSS_PER_TICKER,
    )

sim_w = np.random.dirichlet(np.ones(n), N_SIM_EF)
if ALLOW_SHORTS:
    sim_w = sim_w * np.array([1.0 if POSITION_SIDES.get(t, 'long') == 'long' else -1.0 for t in TICKERS])
sim_stats = np.array([portfolio_stats(w) for w in sim_w])

fig = go.Figure()
fig.add_trace(go.Scatter(x=sim_stats[:,1], y=sim_stats[:,0], mode='markers',
    marker=dict(color=sim_stats[:,2], colorscale='Viridis', size=4,
                showscale=True, colorbar=dict(title='Sharpe')),
    name=f'{N_SIM_EF} random portfolios'))
r_opt, v_opt, s_opt = portfolio_stats(w_sharpe)
fig.add_trace(go.Scatter(x=[v_opt], y=[r_opt], mode='markers+text',
    marker=dict(color='red', size=14, symbol='star'),
    text=['Max Sharpe'], textposition='top right', name='Max Sharpe'))
fig.update_layout(title='Efficient Frontier (gross budget, sentiment-enhanced)',
                  xaxis_title='Volatility', yaxis_title='Return',
                  xaxis_tickformat='.0%', yaxis_tickformat='.0%',
                  template='plotly_dark')
fig.show()

exp = exposure_summary(w_sharpe)
print('\\n📌 Optimised weights:')
for t, w in zip(TICKERS, w_sharpe):
    print(f'  {t}: {w:+.1%}')
print(f'  Return: {r_opt:.2%}  |  Vol: {v_opt:.2%}  |  Sharpe: {s_opt:.2f}')
print(f'  Gross: {exp["gross"]:.2%}  |  Net: {exp["net"]:+.2%}')
print('\\n💡 Re-run Phase 2a to apply optimised/partial weights.')
'''
            set_src(nb["cells"][20], phase3)

    if len(nb["cells"]) > 22 and "aggregate_portfolio_paths" not in src_join(nb["cells"][22]):
        mc = '''if RANDOM_SEED is not None:
    np.random.seed(RANDOM_SEED)

path_by_ticker = {}
raw_by_ticker = {}
for ti, t in enumerate(TICKERS):
    side = 'short' if weights[ti] < 0 else 'long'
    rp = REGIME_PARAMS.get(current_regime, REGIME_PARAMS.get('Neutral ⚪', list(REGIME_PARAMS.values())[1]))
    drift = rp['drift'] / 252 + float(ticker_sentiment.get(t, 0)) * SENTIMENT_DRIFT_NUDGE
    vol_d = max(float(rets[t].std() * np.sqrt(252)), 1e-6) / np.sqrt(252)
    alloc = abs(weights[ti]) * PORT_VAL
    raw = gbm_paths(
        horizon=HORIZON, n_paths=N_PATHS, start_value=alloc,
        drift_daily=drift, vol_daily=vol_d,
        seed=None if RANDOM_SEED is None else RANDOM_SEED + ti,
        side=side,
    )
    raw_by_ticker[t] = raw
    stops_t = stops_by_ticker.get(t, STOPS)
    path_by_ticker[t] = simulate_trailing_stops(raw, stops_t, side='long')

paths_raw = aggregate_portfolio_paths(raw_by_ticker, weights, TICKERS)
paths_stopped = aggregate_portfolio_paths(path_by_ticker, weights, TICKERS)

final_raw = paths_raw[-1]
final_stopped = paths_stopped[-1]
var95_raw = np.percentile(final_raw, 5)
var95_stopped = np.percentile(final_stopped, 5)
days_ax = np.arange(HORIZON + 1)
pct = lambda arr, q: np.percentile(arr, q, axis=1)

stops_sorted = sorted(stops_by_ticker.get(TICKERS[0], STOPS), key=lambda s: s['level'])
triggered = [np.zeros(N_PATHS, dtype=bool) for _ in stops_sorted]

fig = go.Figure()
idx = np.random.choice(N_PATHS, min(PLOT_SAMPLE_PATHS, N_PATHS), replace=False)
for i in idx:
    fig.add_trace(go.Scatter(x=days_ax, y=paths_stopped[:, i],
        line=dict(width=0.5, color='steelblue'), name='With stops',
        legendgroup='s', showlegend=(i == idx[0]), opacity=0.5))

for q, col, dash, lbl in [(95, 'lime', 'dot', '95th (w/ stops)'), (50, 'white', 'solid', 'Median'), (5, 'red', 'dot', '5th (VaR proxy)')]:
    fig.add_trace(go.Scatter(x=days_ax, y=pct(paths_stopped, q), mode='lines',
        line=dict(color=col, width=2, dash=dash), name=lbl))

fig.update_layout(
    title=(f'Monte Carlo — {N_PATHS:,} paths | Per-ticker stops | Regime: {current_regime}'),
    xaxis_title='Trading days', yaxis_title='Portfolio value ($)',
    yaxis_tickformat='$,.0f', template='plotly_dark')
fig.show()

if SHOW_PER_TICKER_MC:
    fig2 = go.Figure()
    for t in TICKERS:
        fig2.add_trace(go.Scatter(x=days_ax, y=path_by_ticker[t][:, 0], name=t, line=dict(width=1)))
    fig2.update_layout(title='Sample per-ticker stopped paths (path 0)', template='plotly_dark')
    fig2.show()

print(f'\\n📉 1-Year Risk Summary | Weights: {PORTFOLIO_WEIGHTING} | Regime: {current_regime}')
print(f'  VaR 95% (stopped aggregate): ${var95_stopped:,.0f}')
'''
        set_src(nb["cells"][22], mc)

    # Insert per-ticker risk markdown + code after phase 2a
    insert_idx = None
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "markdown" and "2a-ii" in src_join(c):
            insert_idx = i
            break

    if insert_idx is not None:
        md = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2a-iii · Per-ticker risk snapshot\n",
                "Position-level metrics (each name traded separately). Portfolio charts below still use the combined `portfolio_rets` series.\n",
            ],
        }
        code = {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "source": [],
        }
        code_text = '''def _ticker_risk_row(t):
    r = rets[t].dropna()
    if r.empty:
        return None
    ann_mu = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe = (ann_mu - RISK_FREE) / ann_vol if ann_vol > 1e-12 else np.nan
    var95 = np.percentile(r, 5)
    cvar95 = r[r <= var95].mean() if (r <= var95).any() else var95
    wi = float(weights[TICKERS.index(t)])
    return {
        'ticker': t,
        'side': 'short' if wi < 0 else 'long',
        'weight': wi,
        'gross': abs(wi),
        'ann_return': ann_mu,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'VaR_95_daily': var95,
        'CVaR_95_daily': cvar95,
    }

rows = [_ticker_risk_row(t) for t in TICKERS]
risk_by_ticker = pd.DataFrame([x for x in rows if x])
display(risk_by_ticker.style.format({
    'weight': '{:+.1%}', 'gross': '{:.1%}',
    'ann_return': '{:.2%}', 'ann_vol': '{:.2%}', 'sharpe': '{:.2f}',
    'VaR_95_daily': '{:.2%}', 'CVaR_95_daily': '{:.2%}',
}))
'''
        set_src(code, code_text)
        # Only insert if not already present
        if not any("2a-iii" in src_join(c) for c in nb["cells"]):
            nb["cells"].insert(insert_idx, code)
            nb["cells"].insert(insert_idx, md)

    # Update phase 2 header markdown
    for c in nb["cells"]:
        if c["cell_type"] == "markdown" and "Phase 2 — Risk Models" in src_join(c):
            set_src(
                c,
                "---\n"
                "## Phase 2 — Risk Models\n"
                "### 2a · Download prices & build the portfolio return series\n"
                "\n"
                "Gross budget: `sum(abs(weight)) = 1`. Negative weights = short.\n"
                "Modes: `'equal'`, `'manual'`, `'partial'` (anchor + optimize rest), `'optimised'`.\n"
                "\n"
                "Portfolio charts (vol, VaR, HMM, EF) use **combined** `portfolio_rets`.\n"
                "Per-ticker tables use individual return series.\n",
            )

    # Trailing stop markdown
    for c in nb["cells"]:
        if c["cell_type"] == "markdown" and "Trailing Stop Configuration" in src_join(c):
            set_src(
                c,
                "## ⚙️ Trailing Stop Configuration\n"
                "Each **ticker** has 3 tranches (⅓ each), with levels from **that ticker's** sentiment.\n"
                "A stop watches drawdown from the running peak (position MTM); shorts use inverted GBM paths.\n"
                "\n"
                "| Mode | How to activate |\n"
                "|---|---|\n"
                "| **Sentiment-derived** *(default)* | `USE_MANUAL = False` — per-ticker after Phase 1 |\n"
                "| **Manual override** | `USE_MANUAL = True` and fill `MANUAL_STOP_LEVELS` / `_FRACTIONS` |\n",
            )

    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Notebook patched:", NB_PATH)


if __name__ == "__main__":
    main()
