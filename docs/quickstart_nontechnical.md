# Non-technical quickstart (Windows / macOS / Linux)

This guide is for people who **don’t want to think about Python environments**. You will:

1) download the project  
2) run **one command**  
3) open the dashboard in your browser

---

## What you need (2 minutes)

- **Internet** (first run downloads Python packages and the FinBERT model)
- **Python 3.12**
  - Windows: install from `python.org` (check “Add python.exe to PATH”), or install from the Microsoft Store
  - macOS: `python3.12` via Homebrew is usually easiest
  - Linux: install `python3.12` from your distro (or `deadsnakes` on Ubuntu)

If you already have Python installed, you can check it with:

- Windows (PowerShell):

```powershell
py -3.12 --version
```

- macOS/Linux (Terminal):

```bash
python3.12 --version
```

---

## Step 1 — Download the project

### Option A (recommended): ZIP download

1. Download the ZIP (from GitHub “Code → Download ZIP” or from a GitHub Release)
2. Unzip it
3. Open the unzipped folder

### Option B: Clone with Git (if you already have Git)

```bash
git clone <REPO_URL_HERE>
cd ai-quant-risk-engine
```

---

## Configure your run (optional)

Edit [configs/run.yaml](../configs/run.yaml) for tickers, `demo` vs `live`, and portfolio weights. See [run_profile.md](run_profile.md).

## Step 2 — Run one command

### Windows (PowerShell)

Open PowerShell **in the project folder**, then run **one** of these:

**Option A — single line (copy/paste once):**

```powershell
Set-ExecutionPolicy -Scope Process Bypass; .\scripts\setup_and_run_windows.ps1
```

**Option B — double-click** `scripts\Run-AQRE.cmd` in File Explorer (no PowerShell typing).

**Option C — two lines** (run line 1, press Enter, then line 2):

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_and_run_windows.ps1
```

Optional flags (override [configs/run.yaml](../configs/run.yaml) only when needed):
- `-MarketSource sample` or `-MarketSource yfinance` (default: use `mode` from run.yaml, usually **live**)
- `-PinDates` (more reproducible dates)
- `-SkipRepro` (skip the long pipeline run and just launch the dashboard)
- `-Legacy` (use the Phase 4 legacy dashboard instead of the Phase 5 platform dashboard)
- `-Api` (also start the FastAPI server after the dashboard)

Examples:

```powershell
.\scripts\setup_and_run_windows.ps1 -MarketSource sample
.\scripts\setup_and_run_windows.ps1 -MarketSource yfinance
.\scripts\setup_and_run_windows.ps1 -SkipRepro
```

### macOS / Linux (Terminal)

Open Terminal **in the project folder**, then run:

```bash
bash scripts/setup_and_run_macos_linux.sh
```

Optional flags (override [configs/run.yaml](../configs/run.yaml) only when needed):
- `--market-source sample` or `--market-source yfinance` (default: use `mode` from run.yaml)
- `--pin-dates`
- `--skip-repro`
- `--legacy`
- `--api`

Examples:

```bash
bash scripts/setup_and_run_macos_linux.sh --market-source sample
bash scripts/setup_and_run_macos_linux.sh --market-source yfinance
bash scripts/setup_and_run_macos_linux.sh --skip-repro
```

---

## Step 3 — Open the dashboard

The command will print a local URL like:
- `http://localhost:8501` (dashboard)
- `http://localhost:8000` (API, if you enabled it)

Open that link in your browser.

In the left sidebar:

- **Portfolio** — weights for your chosen `weighting` mode (`equal`, `manual`, `partial`, `optimised`); see [portfolio_dashboard.md](portfolio_dashboard.md).
- **Signals** — FinBERT scores, trailing stop levels, HMM regime, and VaR 95%; see [signals_dashboard.md](signals_dashboard.md).

---

## What to expect (first run)

- **Time**: often **10–30+ minutes** (depends on your internet + CPU)
- **Downloads**: large Python packages (notably Torch) and the **FinBERT** model
- **Disk**: make sure you have a few GB free (packages + caches + generated artifacts)

The project stores caches inside the repo folder:
- `.cache/huggingface/` (FinBERT / transformers cache)
- `.cache/pip/` (pip download cache)

---

## Troubleshooting

### “Python 3.12 not found”

- Windows: install Python 3.12 and retry. Then run `py -3.12 --version`.
- macOS/Linux: install `python3.12` and retry. Then run `python3.12 --version`.

### “The term 'dvc' is not recognized” / “No module named aqre”

The script installs everything into a local virtual environment (`.venv312/`). The dashboard is launched via:

```powershell
.\.venv312\Scripts\python.exe -m src.cli dashboard
```

(Older docs/scripts used `python -m aqre`, which does not work — `aqre` is a console script, not a Python module.)

### DVC error: `metrics\...` is already tracked by SCM

If CI or `dvc repro` fails with *"output 'metrics/sentiment' is already tracked by SCM"*, Git is tracking folders that DVC generates. Run once from the repo root (PowerShell):

```powershell
git rm -r --cached metrics/sentiment metrics/platform metrics/research_backtests metrics/research_compare metrics/research_evaluation metrics/research_reports metrics/research_scenarios metrics/research_simulations metrics/research_stress 2>$null
git commit -m "Stop tracking DVC-generated metrics folders"
```

Then re-run your pipeline. These paths are listed in `.gitignore` and should never be committed.

### FinBERT download is slow / fails

This is usually a network/proxy issue. Try:
- switching networks (home vs corporate)
- retrying later

You can still validate installation quickly with `--skip-repro` (dashboard will load but may warn that artifacts are missing).

### yfinance / SSL errors (live mode)

The pipeline tries several Yahoo download strategies automatically (default client, then SSL-relaxed fallback). If prepare still fails with certificate errors, retry after:

```powershell
.venv312\Scripts\pip install certifi curl_cffi
```

Or one session only (dev): `$env:YFINANCE_SSL_VERIFY = "0"` then re-run the setup script.
