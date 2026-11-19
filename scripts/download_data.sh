#!/usr/bin/env bash
# Download the raw Kaggle datasets into data/raw/.
#
# Prerequisites (you do these once, yourself):
#   1. kaggle.com > Settings > API. Either:
#        a. "Generate New Token" (recommended) -> save the KGAT_... value to
#           ~/.kaggle/access_token, or export KAGGLE_API_TOKEN=KGAT_...
#        b. "Create Legacy API Key" -> downloads kaggle.json; move it to
#           ~/.kaggle/kaggle.json and chmod 600 it.
#   2. Accept the M5 competition rules. The download returns HTTP 403 until you
#      have, and only the account holder can accept them:
#      https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules
#
# Raw files are gitignored and never committed.
set -euo pipefail

RAW_DIR="${RAW_DIR:-data/raw}"
M5_DIR="$RAW_DIR/m5"
DATACO_DIR="$RAW_DIR/dataco"

if [ ! -f "$HOME/.kaggle/access_token" ] \
   && [ ! -f "$HOME/.kaggle/kaggle.json" ] \
   && [ -z "${KAGGLE_API_TOKEN:-}" ] \
   && [ -z "${KAGGLE_USERNAME:-}" ]; then
  echo "ERROR: no Kaggle credentials found." >&2
  echo "       Expected ~/.kaggle/access_token or ~/.kaggle/kaggle.json" >&2
  echo "       (see the header of this script)." >&2
  exit 1
fi

KAGGLE_BIN="$(command -v kaggle || true)"
if [ -z "$KAGGLE_BIN" ] && [ -x ".venv/bin/kaggle" ]; then
  KAGGLE_BIN=".venv/bin/kaggle"
fi
if [ -z "$KAGGLE_BIN" ]; then
  echo "Installing the kaggle CLI into .venv (dev-only; not in requirements.txt) ..."
  # >= 1.8.0 is required for the KGAT_ API tokens.
  .venv/bin/pip install --quiet --upgrade "kaggle>=1.8.0"
  KAGGLE_BIN=".venv/bin/kaggle"
fi

mkdir -p "$M5_DIR" "$DATACO_DIR"

echo "==> M5 Forecasting - Accuracy (competition)"
if ! "$KAGGLE_BIN" competitions download -c m5-forecasting-accuracy -p "$M5_DIR" --force; then
  echo >&2
  echo "ERROR: the M5 download failed. The usual cause is that the competition" >&2
  echo "       rules have not been accepted on this Kaggle account. Accept them at" >&2
  echo "       https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules" >&2
  echo "       and run this script again." >&2
  exit 1
fi
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
