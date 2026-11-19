#!/usr/bin/env bash
# Download the raw Kaggle datasets into data/raw/.
#
# Prerequisites (you do these once, yourself):
#   1. kaggle.com > Settings > API > Create New Token  -> downloads kaggle.json
#   2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
#   3. Accept the M5 competition rules:
#      https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules
#
# Raw files are gitignored and never committed.
set -euo pipefail

RAW_DIR="${RAW_DIR:-data/raw}"
M5_DIR="$RAW_DIR/m5"
DATACO_DIR="$RAW_DIR/dataco"

if [ ! -f "$HOME/.kaggle/kaggle.json" ] && [ -z "${KAGGLE_USERNAME:-}" ]; then
  echo "ERROR: no Kaggle credentials found." >&2
  echo "       Expected ~/.kaggle/kaggle.json (see the header of this script)." >&2
  exit 1
fi

KAGGLE_BIN="$(command -v kaggle || true)"
if [ -z "$KAGGLE_BIN" ] && [ -x ".venv/bin/kaggle" ]; then
  KAGGLE_BIN=".venv/bin/kaggle"
fi
if [ -z "$KAGGLE_BIN" ]; then
  echo "Installing the kaggle CLI into .venv (dev-only; not in requirements.txt) ..."
  .venv/bin/pip install --quiet kaggle
  KAGGLE_BIN=".venv/bin/kaggle"
fi

mkdir -p "$M5_DIR" "$DATACO_DIR"

echo "==> M5 Forecasting - Accuracy (competition)"
"$KAGGLE_BIN" competitions download -c m5-forecasting-accuracy -p "$M5_DIR" --force
unzip -o -q "$M5_DIR"/m5-forecasting-accuracy.zip -d "$M5_DIR"
rm -f "$M5_DIR"/m5-forecasting-accuracy.zip

echo "==> DataCo Smart Supply Chain (dataset)"
"$KAGGLE_BIN" datasets download -d shashwatwork/dataco-smart-supply-chain-for-big-data-analysis \
  -p "$DATACO_DIR" --force --unzip

echo
echo "Downloaded into $RAW_DIR:"
find "$RAW_DIR" -maxdepth 2 -name '*.csv' -exec ls -lh {} \; | awk '{print "  " $5 "  " $9}'
echo
echo "Now run:  make ingest-real"
