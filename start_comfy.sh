#!/bin/bash
# =============================================================
# motion-control-api — ComfyUI supervisor on :8188
# Same flock-guarded restart-on-exit pattern as ai-gen-api-v2.
# =============================================================
LOG_SETUP="/workspace/comfy_setup.log"
LOG_OUT="/workspace/comfyui.log"
ARGS_FILE="/workspace/runpod-slim/comfyui_args.txt"
PORT=8188
FIXED_ARGS=(--listen 0.0.0.0 --port "$PORT" --enable-cors-header)

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_SETUP"; }

# ─── single-instance guard ───
exec 9>/var/lock/motion-comfy.lock
if ! flock -n 9; then
  log "start_comfy.sh: another supervisor already running — exiting"
  exit 0
fi

# Truncate old logs on restart (cap at last 500 lines each)
tail -500 "$LOG_SETUP" > "${LOG_SETUP}.tmp" 2>/dev/null && mv "${LOG_SETUP}.tmp" "$LOG_SETUP"
tail -500 "$LOG_OUT"   > "${LOG_OUT}.tmp"   2>/dev/null && mv "${LOG_OUT}.tmp"   "$LOG_OUT"

if [ ! -f /workspace/api/config.env ]; then
  log "ERROR: /workspace/api/config.env missing — setup.sh did not complete"
  exit 1
fi
set -a
source /workspace/api/config.env
set +a

if [ -z "$PYTHON" ] || [ -z "$COMFY_ROOT" ]; then
  log "ERROR: PYTHON or COMFY_ROOT not set in config.env"
  exit 1
fi
if [ ! -x "$PYTHON" ]; then
  log "ERROR: PYTHON ($PYTHON) not executable"
  exit 1
fi
if [ ! -f "$COMFY_ROOT/main.py" ]; then
  log "ERROR: $COMFY_ROOT/main.py not found"
  exit 1
fi

# Free :PORT if a stale ComfyUI is holding it (mirrors ai-gen-api-v2 logic
# — kill the port owner instead of pkill-by-argv-regex).
STALE_PID=$(netstat -tlnp 2>/dev/null | awk -v p=":$PORT\$" '$4 ~ p {split($7, a, "/"); print a[1]; exit}')
if [ -n "$STALE_PID" ] && [ "$STALE_PID" != "-" ]; then
  log "Freeing :$PORT (stale owner PID=$STALE_PID)"
  kill "$STALE_PID" 2>/dev/null || true
  for _ in 1 2 3 4 5; do kill -0 "$STALE_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$STALE_PID" 2>/dev/null || true
fi

# Extra args (one per line, # comments ok)
EXTRA_ARGS=()
if [ -f "$ARGS_FILE" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    trimmed="${line#"${line%%[![:space:]]*}"}"
    [ -z "$trimmed" ] && continue
    [[ "$trimmed" =~ ^# ]] && continue
    args=($trimmed)
    EXTRA_ARGS+=("${args[@]}")
  done < "$ARGS_FILE"
fi
log "ComfyUI extra args: ${EXTRA_ARGS[*]:-(none)}"

cd "$COMFY_ROOT" || exit 1
log "Starting ComfyUI on port $PORT..."
while true; do
  "$PYTHON" main.py "${FIXED_ARGS[@]}" "${EXTRA_ARGS[@]}" >> "$LOG_OUT" 2>&1
  EXIT_CODE=$?
  log "ComfyUI exited with code $EXIT_CODE — restarting in 5s..."
  sleep 5
done
