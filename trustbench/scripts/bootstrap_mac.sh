#!/usr/bin/env bash
set -e

echo ">>> Checking Homebrew..."
if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew not found. Please install from https://brew.sh/ and re-run."
  exit 1
fi

echo ">>> Creating Python venv .venv ..."
python3 -m venv .venv
source .venv/bin/activate

echo ">>> Upgrading pip ..."
python -m pip install --upgrade pip

echo ">>> Installing Python dependencies ..."
pip install -r requirements.txt

echo ">>> Checking Ollama ..."
if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama not found. Installing via Homebrew ..."
  brew install ollama
fi

echo ">>> Pulling model llama3.2:1b (this may take a few minutes the first time) ..."
ollama pull llama3.2:1b || true

echo ">>> Done. Activate venv and run:"
echo "source .venv/bin/activate"
echo "python trustbench.py --backend ollama --subset 150 --metric ask"
