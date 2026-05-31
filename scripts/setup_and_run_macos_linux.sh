#!/usr/bin/env bash
set -euo pipefail

market_source=""
pin_dates="0"
skip_repro="0"
legacy="0"
api="0"

usage() {
  cat <<'EOF'
One-command bootstrap for non-technical users (macOS/Linux).

Usage:
  bash scripts/setup_and_run_macos_linux.sh

Options:
  --market-source sample|yfinance   (optional; default: configs/run.yaml mode)
  --pin-dates                      (sets MARKET_PIN_DATES=1)
  --skip-repro                     (skip `dvc repro`)
  --legacy                         (launch legacy Phase 4 dashboard)
  --api                            (also start FastAPI server after dashboard)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --market-source)
      market_source="${2:-}"; shift 2;;
    --pin-dates)
      pin_dates="1"; shift;;
    --skip-repro)
      skip_repro="1"; shift;;
    --legacy)
      legacy="1"; shift;;
    --api)
      api="1"; shift;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1;;
  esac
done

if [[ ! -f "pyproject.toml" ]]; then
  echo "Run this script from the repo root (folder containing pyproject.toml)." >&2
  exit 1
fi

if [[ "$market_source" != "" && "$market_source" != "sample" && "$market_source" != "yfinance" ]]; then
  echo "--market-source must be 'sample' or 'yfinance' (got: $market_source)" >&2
  exit 1
fi

echo ""
echo "== AI Quant Risk Engine — setup =="

# Resolve Python 3.12
py312=""
if command -v python3.12 >/dev/null 2>&1; then
  py312="python3.12"
elif command -v python3 >/dev/null 2>&1; then
  if python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:2]==(3,12) else 1)' >/dev/null 2>&1; then
    py312="python3"
  fi
fi

if [[ -z "$py312" ]]; then
  echo "Python 3.12 is required. Please install python3.12, then retry." >&2
  exit 1
fi

venv_dir=".venv312"
venv_py="$venv_dir/bin/python"
if [[ ! -x "$venv_py" ]]; then
  echo ""
  echo "== Creating virtual environment =="
  "$py312" -m venv "$venv_dir"
fi

if [[ ! -x "$venv_py" ]]; then
  echo "Virtual environment creation failed: $venv_py not found." >&2
  exit 1
fi

# Ensure `python` inside DVC stages resolves to the venv interpreter
export PATH="$(cd "$venv_dir/bin" && pwd):$PATH"

# Cache dirs inside repo (so reruns are fast)
cache_root=".cache"
hf_home="$cache_root/huggingface"
pip_cache="$cache_root/pip"
mkdir -p "$hf_home" "$pip_cache"

export HF_HOME="$(cd "$hf_home" && pwd)"
export TRANSFORMERS_CACHE="$HF_HOME"
export PIP_CACHE_DIR="$(cd "$pip_cache" && pwd)"

if [[ -n "$market_source" ]]; then
  export MARKET_SOURCE="$market_source"
fi
if [[ "$pin_dates" == "1" ]]; then
  export MARKET_PIN_DATES="1"
fi

echo ""
echo "== Installing dependencies (this can take a while) =="
"$venv_py" -m pip install --upgrade pip wheel
"$venv_py" -m pip install 'setuptools>=68,<82'
"$venv_py" -m pip install -e ".[dev,market,risk,research,dashboard,platform]"

if [[ "$skip_repro" != "1" ]]; then
  echo ""
  echo "== Prepare + full pipeline (configs/run.yaml) =="
  profile_args=(run profile --dashboard)
  if [[ "$legacy" == "1" ]]; then
    profile_args+=(--legacy)
  fi
  "$venv_py" -m src.cli "${profile_args[@]}"
else
  echo ""
  echo "== Prepare only (--skip-repro) =="
  "$venv_py" -m src.cli prepare
  echo ""
  echo "== Launching dashboard =="
  if [[ "$legacy" == "1" ]]; then
    "$venv_py" -m src.cli dashboard --legacy
  else
    "$venv_py" -m src.cli dashboard
  fi
fi

if [[ "$api" == "1" ]]; then
  echo ""
  echo "== Launching API (FastAPI) =="
  "$venv_py" -m src.cli api serve
fi

