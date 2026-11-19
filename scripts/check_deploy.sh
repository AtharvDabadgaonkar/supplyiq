#!/usr/bin/env bash
# Boot the app headless in a runtime-only venv and fail if it errors or dies.
#
# This is the local stand-in for a Streamlit Community Cloud deploy: only the
# pinned requirements.txt is installed, so a dev-only import or an unpinned
# dependency shows up here rather than in a deploy log.
set -uo pipefail

VENV="${1:-.deploycheck/venv}"
LOG=".deploycheck/boot.log"
SECONDS_TO_RUN="${SECONDS_TO_RUN:-20}"

# Pick a free port rather than assuming one is available.
PORT="${PORT:-$("$VENV/bin/python" - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)}"

mkdir -p "$(dirname "$LOG")"
echo "using port $PORT"

"$VENV/bin/streamlit" run streamlit_app.py \
  --server.headless true \
  --server.port "$PORT" \
  --browser.gatherUsageStats false \
  >"$LOG" 2>&1 &
PID=$!

cleanup() { kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; }
trap cleanup EXIT

for _ in $(seq 1 "$SECONDS_TO_RUN"); do
  sleep 1
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "FAIL: the app exited during startup." >&2
    cat "$LOG" >&2
    exit 1
  fi
done

# A Streamlit page that raises still leaves the server up, so scan the log too.
if grep -qiE "traceback|modulenotfounderror|importerror|streamlit *\. *errors" "$LOG"; then
  echo "FAIL: errors in the startup log:" >&2
  cat "$LOG" >&2
  exit 1
fi

# Confirm it is actually serving, not just alive.
if ! "$VENV/bin/python" - "$PORT" <<'PY'
import sys, urllib.request
url = f"http://127.0.0.1:{sys.argv[1]}/_stcore/health"
with urllib.request.urlopen(url, timeout=10) as r:
    body = r.read().decode().strip()
print(f"health endpoint: {r.status} {body}")
sys.exit(0 if r.status == 200 else 1)
PY
then
  echo "FAIL: the app did not answer its health endpoint." >&2
  cat "$LOG" >&2
  exit 1
fi

echo "PASS: app ran ${SECONDS_TO_RUN}s on runtime deps only, served health, no startup errors."
echo "----- startup log -----"
cat "$LOG"
