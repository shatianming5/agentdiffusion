#!/bin/bash
set -Eeuo pipefail

OUT_DIR="data/external/nasdaq_itch"
BASE="https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH"

FILES=(
  "01302019.NASDAQ_ITCH50.gz"
  "01302020.NASDAQ_ITCH50.gz"
  "03272019.NASDAQ_ITCH50.gz"
  "07302019.NASDAQ_ITCH50.gz"
  "08302019.NASDAQ_ITCH50.gz"
  "10302019.NASDAQ_ITCH50.gz"
  "12302019.NASDAQ_ITCH50.gz"
  "S071321-v50.txt.gz"
  "S081321-v50.txt.gz"
  "S101819-v50.txt.gz"
)

mkdir -p "${OUT_DIR}"

for F in "${FILES[@]}"; do
  OUT="${OUT_DIR}/${F}"
  if [ -f "$OUT" ]; then
    SIZE=$(du -h "$OUT" | cut -f1)
    echo "SKIP: $F (exists, ${SIZE})"
    continue
  fi
  echo "Downloading $F ..."
  if curl -f --progress-bar -o "$OUT" "${BASE}/${F}"; then
    SIZE=$(du -h "$OUT" | cut -f1)
    echo "OK: $F (${SIZE})"
  else
    rm -f "$OUT"
    echo "FAILED: $F"
  fi
done

echo ""
echo "=== NASDAQ ITCH Downloads ==="
ls -lh ${OUT_DIR}/*.gz 2>/dev/null
echo ""
du -sh ${OUT_DIR}/
