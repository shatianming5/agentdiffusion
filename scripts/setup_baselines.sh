#!/bin/bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."

echo "============================================================"
echo "  Setup baseline models for comparison"
echo "============================================================"

# --- TRADES / DeepMarket ---
if [ ! -d "vendor/DeepMarket" ]; then
    echo "=== Cloning DeepMarket (TRADES) ==="
    git clone https://github.com/LeonardoBerti00/DeepMarket.git vendor/DeepMarket
else
    echo "=== DeepMarket already cloned ==="
fi

# --- LOB-Bench ---
echo "=== Installing lob_bench ==="
.venv/bin/pip install lob_bench 2>/dev/null | tail -1 || echo "lob_bench install failed (may need Python>=3.9)"

echo "=== Done ==="
echo "Baselines available in vendor/"
ls -d vendor/*/
