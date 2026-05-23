#!/bin/bash
# =============================================================
# motion-control-api — FastAPI supervisor on :7860
#
# Fetches latest main.py / workflows.py from GitHub at the START of
# every loop iteration so that any commit on main propagates to the
# running pod on the next uvicorn restart (matches ai-gen-api-v2's
# `fetch_api_code` + loop pattern).
# =============================================================
LOG_SETUP="/workspace/api_setup.log"
LOG_OUT="/workspace/api.log"
PORT=7860

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_SETUP"; }

# Single-instance guard
exec 9>/var/lock/motion-api.lock
if ! flock -n 9; then
  log "start_api.sh: another supervisor already running — exiting"
  exit 0
fi

tail -500 "$LOG_SETUP" > "${LOG_SETUP}.tmp" 2>/dev/null && mv "${LOG_SETUP}.tmp" "$LOG_SETUP"
tail -500 "$LOG_OUT"   > "${LOG_OUT}.tmp"   2>/dev/null && mv "${LOG_OUT}.tmp"   "$LOG_OUT"

if [ ! -f /workspace/api/config.env ]; then
  log "ERROR: /workspace/api/config.env missing — setup.sh did not complete"
  exit 1
fi
set -a
source /workspace/api/config.env
set +a

if [ -z "$PYTHON" ] || [ -z "$API_DIR" ] || [ -z "$API_REPO" ]; then
  log "ERROR: PYTHON / API_DIR / API_REPO missing from config.env"
  exit 1
fi

# Free :PORT if a stale uvicorn / install-status-server is holding it
STALE_PID=$(netstat -tlnp 2>/dev/null | awk -v p=":$PORT\$" '$4 ~ p {split($7, a, "/"); print a[1]; exit}')
if [ -n "$STALE_PID" ] && [ "$STALE_PID" != "-" ]; then
  log "Freeing :$PORT (stale owner PID=$STALE_PID)"
  kill "$STALE_PID" 2>/dev/null || true
  for _ in 1 2 3 4 5; do kill -0 "$STALE_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$STALE_PID" 2>/dev/null || true
fi

fetch_api_code() {
  # Re-fetch on every loop iteration so a fresh push to main propagates
  # to the next uvicorn launch. Cache-busted via ?cb=<unix_ts> to dodge
  # any stale GitHub raw CDN edge.
  local cb; cb=$(date +%s)
  for f in main.py workflows.py safety.py setup.py; do
    local url="$API_REPO/$f?cb=$cb"
    local tmp="$API_DIR/$f.fetch"
    if wget -q "$url" -O "$tmp" && [ -s "$tmp" ]; then
      mv "$tmp" "$API_DIR/$f"
    else
      rm -f "$tmp"
    fi
  done
}

cd "$API_DIR" || exit 1
log "Starting FastAPI supervisor on port $PORT..."
while true; do
  fetch_api_code
  "$PYTHON" -m uvicorn main:app --host 0.0.0.0 --port "$PORT" >> "$LOG_OUT" 2>&1
  EXIT_CODE=$?
  log "uvicorn exited with code $EXIT_CODE — restarting in 5s..."
  sleep 5
done
