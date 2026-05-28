<#
One-command bootstrap for non-technical users (Windows).

What it does:
- checks Python 3.12
- creates .venv312 if missing
- installs this repo + extras needed for full pipeline + Phase 5 dashboard
- runs `dvc repro` (optional)
- launches the dashboard (Phase 5 by default)

Usage (PowerShell):
  Set-ExecutionPolicy -Scope Process Bypass
  .\scripts\setup_and_run_windows.ps1

Optional:
  .\scripts\setup_and_run_windows.ps1 -MarketSource yfinance -PinDates
  .\scripts\setup_and_run_windows.ps1 -SkipRepro
  .\scripts\setup_and_run_windows.ps1 -Legacy
  .\scripts\setup_and_run_windows.ps1 -Api
#>

[CmdletBinding()]
param(
  [ValidateSet("sample", "yfinance")]
  [string]$MarketSource = "sample",
  [switch]$PinDates,
  [switch]$SkipRepro,
  [switch]$Legacy,
  [switch]$Api
)

$ErrorActionPreference = "Stop"

function Write-Section([string]$Text) {
  Write-Host ""
  Write-Host "== $Text =="
}

Write-Section 'AI Quant Risk Engine - setup'

if (-not (Test-Path -Path ".\pyproject.toml")) {
  throw "Run this script from the repo root (folder containing pyproject.toml)."
}

# Resolve Python 3.12
$python = $null
try {
  $python = (Get-Command py -ErrorAction Stop).Source
} catch {
  $python = $null
}

function Get-Python312Exe {
  if ($python -ne $null) {
    try {
      & py -3.12 -c "import sys; assert sys.version_info[:2]==(3,12)" | Out-Null
      return "py -3.12"
    } catch {
      # fall through
    }
  }
  try {
    & python --version | Out-Null
    & python -c "import sys; assert sys.version_info[:2]==(3,12)" | Out-Null
    return "python"
  } catch {
    return $null
  }
}

$py312 = Get-Python312Exe
if ($py312 -eq $null) {
  throw 'Python 3.12 is required. Install Python 3.12, then retry. (Test: py -3.12 --version)'
}

# Create venv
$venvDir = ".venv312"
$venvPy = Join-Path $venvDir "Scripts\python.exe"
if (-not (Test-Path -Path $venvPy)) {
  Write-Section "Creating virtual environment"
  if ($py312 -eq "py -3.12") {
    & py -3.12 -m venv $venvDir
  } else {
    & python -m venv $venvDir
  }
}

if (-not (Test-Path -Path $venvPy)) {
  throw "Virtual environment creation failed: $venvPy not found."
}

# Cache dirs inside repo (so reruns are fast)
$cacheRoot = ".cache"
$hfHome = Join-Path $cacheRoot "huggingface"
$pipCache = Join-Path $cacheRoot "pip"
New-Item -ItemType Directory -Force -Path $hfHome | Out-Null
New-Item -ItemType Directory -Force -Path $pipCache | Out-Null

$env:HF_HOME = (Resolve-Path $hfHome).Path
$env:TRANSFORMERS_CACHE = (Resolve-Path $hfHome).Path
$env:PIP_CACHE_DIR = (Resolve-Path $pipCache).Path

# Ensure `python` inside DVC stages resolves to the venv interpreter
$venvScripts = (Resolve-Path (Join-Path $venvDir 'Scripts')).Path
$env:Path = $venvScripts + ';' + $env:Path

# Optional session overrides (configs/run.yaml is the default source of truth)
if ($MarketSource) { $env:MARKET_SOURCE = $MarketSource }
if ($PinDates) { $env:MARKET_PIN_DATES = "1" }

Write-Section "Installing dependencies (this can take a while)"
& $venvPy -m pip install --upgrade pip wheel
# torch pins setuptools<82; avoid upgrading setuptools past that in the bootstrap script
& $venvPy -m pip install 'setuptools>=68,<82'
& $venvPy -m pip install -e '.[dev,market,risk,research,dashboard,platform]'

if (-not $SkipRepro) {
  Write-Section "Prepare + full pipeline (configs/run.yaml)"
  $profileArgs = @("run", "profile", "--dashboard")
  if ($Legacy) { $profileArgs += "--legacy" }
  & $venvPy -m src.cli @profileArgs
} else {
  Write-Section "Prepare only (SkipRepro)"
  & $venvPy -m src.cli prepare
  Write-Section "Launching dashboard"
  if ($Legacy) {
    & $venvPy -m src.cli dashboard --legacy
  } else {
    & $venvPy -m src.cli dashboard
  }
}

if ($Api) {
  Write-Section 'Launching API (FastAPI)'
  & $venvPy -m src.cli api serve
}

