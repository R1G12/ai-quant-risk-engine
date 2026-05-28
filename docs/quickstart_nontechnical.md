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

## Step 2 — Run one command

### Windows (PowerShell)

Open PowerShell **in the project folder**, then run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_and_run_windows.ps1
```

Optional flags:
- `-MarketSource sample` (default) or `-MarketSource yfinance`
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

Optional flags:
- `--market-source sample` (default) or `--market-source yfinance`
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

### “The term 'dvc' is not recognized” / “aqre not found”

The script installs everything into a local virtual environment (`.venv312/`). If the install step failed, scroll up to find the error and re-run the command.\n+
### FinBERT download is slow / fails

This is usually a network/proxy issue. Try:
- switching networks (home vs corporate)
- retrying later\n+
You can still validate installation quickly with `--skip-repro` (dashboard will load but may warn that artifacts are missing).\n+
### yfinance errors (only if using `yfinance` market source)

Yahoo sometimes blocks requests or SSL cert validation can fail on some machines.\n+
Try running with the default `sample` market source first (fast install sanity check), then retry `yfinance`.\n+
