#!/bin/bash
# =============================================================
# motion-control-api — FastAPI supervisor on :7860
#
# CHANGED from the original wget-on-every-loop fetch: the supervisor
# no longer auto-fetches main.py/workflows.py from GitHub raw at the
# start of each iteration. That mechanism was racing against the
# setup.py shim — the shim would shutil.copy2 the freshly cloned
# files in, then the supervisor's wget would overwrite them with
# stale CDN content on the very next uvicorn restart, so /motion
# kept running the old code.
#
# New deploy flow: code refreshes happen ONLY through the setup.py
# shim, triggered by POST /admin/install-comfy-node. The shim does
# `shutil.copy2(src=local_clone, dst=/workspace/api/)` — local IO,
# no CDN involved — and SIGKILLs uvicorn. The supervisor below just
# restarts whatever main.py is on disk. (The shim's git clone /
# git pull at install-time is the cache-busting source of truth.)
#
# Manual fallback: `/admin/exec-shell` can curl-and-replace files
# directly when we need an out-of-band patch.
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

if [ -z "$PYTHON" ] || [ -z "$API_DIR" ]; then
  log "ERROR: PYTHON / API_DIR missing from config.env"
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

cd "$API_DIR" || exit 1
log "Starting FastAPI supervisor on port $PORT (shim-only deploy mode)..."
while true; do
  "$PYTHON" -m uvicorn main:app --host 0.0.0.0 --port "$PORT" >> "$LOG_OUT" 2>&1
  EXIT_CODE=$?
  log "uvicorn exited with code $EXIT_CODE — restarting in 5s..."
  sleep 5
done
