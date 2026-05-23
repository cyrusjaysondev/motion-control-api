#!/bin/bash
# =============================================================
# motion-control-api — Setup
# Wan 2.2 Animate 14B (character-driven motion transfer)
#
# Works with RunPod ComfyUI template (runpod/comfyui:latest).
# Same volume layout as ai-gen-api-v2 ComfyUI root:
#   /workspace/runpod-slim/ComfyUI/
#
# Required env vars (set in RunPod template):
#   HF_TOKEN = your Hugging Face token (for gated downloads if any)
# =============================================================

LOG="/workspace/motion_setup.log"
log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }

# ─────────────────────────────────────────────
# Single-instance lock — prevents concurrent installs colliding on
# aria2 partial files or racing supervisor launches.
# ─────────────────────────────────────────────
exec 8>/var/lock/motion-control-api-setup.lock
if ! flock -n 8; then
  log "setup.sh: another setup already in progress — exiting"
  exit 0
fi

API_REPO="https://raw.githubusercontent.com/cyrusjaysondev/motion-control-api/main"

# ─────────────────────────────────────────────
# Status server on :7860 so the proxy returns 503 + a useful JSON
# body during install, not a Cloudflare 502. start_api.sh's port
# claim logic supersedes this once uvicorn is ready.
# ─────────────────────────────────────────────
STATUS_PID_FILE=/var/run/motion-control-api-status.pid
if [ -f "$STATUS_PID_FILE" ]; then
  kill "$(cat "$STATUS_PID_FILE" 2>/dev/null)" 2>/dev/null || true
  rm -f "$STATUS_PID_FILE"
  sleep 0.5
fi

if pgrep -xf "bash /workspace/start_api.sh" >/dev/null 2>&1; then
  log "start_api.sh supervisor already running — skipping status server"
elif netstat -tln 2>/dev/null | grep -q ":7860 "; then
  log ":7860 already bound — skipping status server"
else
  cat > /tmp/motion-status.py <<'PYEOF'
import http.server, socketserver, json, os, subprocess, signal, sys
socketserver.ThreadingTCPServer.allow_reuse_address = True

def recent_log():
    try:
        return subprocess.check_output(
            ['tail', '-40', '/workspace/motion_setup.log'],
            text=True, timeout=2
        ).splitlines()[-30:]
    except Exception:
        return []

class H(http.server.BaseHTTPRequestHandler):
    def _send(self, code, body):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        payload = {
            "status": "installing",
            "message": (
                "motion-control-api is still setting up. First deploy "
                "downloads ~30 GB of Wan 2.2 Animate model weights "
                "(~5-15 min on warm HF CDN). /healthz returns HTTP 200 "
                "once uvicorn is bound."
            ),
            "pod_id": os.environ.get('RUNPOD_POD_ID', 'unknown'),
            "hint": "tail -f /workspace/motion_setup.log",
            "recent_log": recent_log(),
        }
        self._send(503, json.dumps(payload, indent=2).encode())

signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
try:
    with socketserver.ThreadingTCPServer(("0.0.0.0", 7860), H) as srv:
        srv.serve_forever()
except OSError:
    sys.exit(0)
PYEOF
  setsid nohup python3 /tmp/motion-status.py </dev/null >/dev/null 2>&1 8>&- &
  echo $! > "$STATUS_PID_FILE"
  sleep 0.5
  if kill -0 "$(cat "$STATUS_PID_FILE")" 2>/dev/null; then
    log "Status server bound :7860"
  else
    log "WARN: status server failed to start (port taken?)"
    rm -f "$STATUS_PID_FILE"
  fi
fi

# ─────────────────────────────────────────────
# HF Token (Wan 2.2 Animate may not be gated, but include hook for
# future gated variants and faster authenticated CDN routing)
# ─────────────────────────────────────────────
TOKEN="${HF_TOKEN:-$HUGGING_FACE_HUB_TOKEN}"
[ -n "$TOKEN" ] && log "HF_TOKEN present (authenticated CDN routing)" || \
  log "HF_TOKEN unset — public CDN only (slower)"

# ─────────────────────────────────────────────
# Auto-detect ComfyUI location + Python
# ─────────────────────────────────────────────
if [ -d "/workspace/runpod-slim/ComfyUI" ]; then
  COMFY_ROOT="/workspace/runpod-slim/ComfyUI"
elif [ -d "/workspace/ComfyUI" ]; then
  COMFY_ROOT="/workspace/ComfyUI"
else
  COMFY_ROOT=$(find /workspace -name "main.py" -path "*/ComfyUI/*" -exec dirname {} \; 2>/dev/null | head -1)
  if [ -z "$COMFY_ROOT" ]; then
    log "ERROR: ComfyUI not found. Exiting."
    exit 1
  fi
fi

if [ -f "$COMFY_ROOT/.venv-cu128/bin/python" ]; then
  PYTHON="$COMFY_ROOT/.venv-cu128/bin/python"
  PIP="$COMFY_ROOT/.venv-cu128/bin/pip"
elif [ -f "/opt/venv/bin/python" ]; then
  PYTHON="/opt/venv/bin/python"
  PIP="/opt/venv/bin/pip"
else
  PYTHON=$(which python3)
  PIP=$(which pip3)
fi

MODELS="$COMFY_ROOT/models"
NODES="$COMFY_ROOT/custom_nodes"

log "=========================================="
log "motion-control-api Setup Started"
log "Pod ID: ${RUNPOD_POD_ID:-unknown}"
log "ComfyUI: $COMFY_ROOT"
log "Python: $PYTHON"
log "=========================================="

# ─────────────────────────────────────────────
# 1. Pip dependencies
# ─────────────────────────────────────────────
log "[1/4] Installing pip dependencies..."
$PIP install -q fastapi uvicorn[standard] httpx websockets python-multipart pillow 2>&1 | tail -1
if ! command -v aria2c >/dev/null 2>&1; then
  log "  Installing aria2 for parallel downloads..."
  apt-get update -qq 2>&1 | tail -1
  apt-get install -y -qq aria2 2>&1 | tail -1
fi
log "  Done"

# ─────────────────────────────────────────────
# 2. Custom nodes
#    - kijai/ComfyUI-WanVideoWrapper: production node for Wan 2.2 Animate
#    - Kosinkadink/ComfyUI-VideoHelperSuite: VHS_LoadVideo + VHS_VideoCombine
#    - Fannovel16/comfyui_controlnet_aux: DWPose if we need preprocessing
# ─────────────────────────────────────────────
log "[2/4] Installing ComfyUI custom nodes..."
mkdir -p "$NODES"

install_node() {
  local url="$1"; local name="$2"
  local dir="$NODES/$name"
  if [ -d "$dir/.git" ]; then
    log "  $name: pull..."
    (cd "$dir" && git pull --ff-only 2>&1 | tail -1)
  else
    log "  $name: clone..."
    git clone --depth 1 "$url" "$dir" 2>&1 | tail -1
  fi
  if [ -f "$dir/requirements.txt" ]; then
    log "  $name: pip install -r requirements.txt..."
    $PIP install -q -r "$dir/requirements.txt" 2>&1 | tail -1
  fi
}

install_node "https://github.com/kijai/ComfyUI-WanVideoWrapper" "ComfyUI-WanVideoWrapper"
install_node "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite" "ComfyUI-VideoHelperSuite"
install_node "https://github.com/Fannovel16/comfyui_controlnet_aux" "comfyui_controlnet_aux"
log "  Done"

# ─────────────────────────────────────────────
# 3. Download Wan 2.2 Animate model weights (~30 GB)
#    Source: Kijai/WanVideo_comfy on HuggingFace — pre-packaged for
#    the ComfyUI-WanVideoWrapper node layout.
# ─────────────────────────────────────────────
log "[3/4] Downloading Wan 2.2 Animate models..."
mkdir -p "$MODELS/diffusion_models" "$MODELS/vae" "$MODELS/text_encoders"

ARIA2_INPUT="/tmp/motion-downloads.txt"
HF="https://huggingface.co"

# Build the aria2 input file. Each entry is:
#   <url>
#     dir=<target_directory>
#     out=<filename>
# `aria2c -i` reads this format. -x 16 -s 16 = 16 connections per file.
#
# Files (per Comfy's official Wan 2.2 Animate workflow docs at
# docs.comfy.org/tutorials/video/wan/wan2-2-animate):
#   - Diffusion model: Comfy-Org/Wan_2.2_ComfyUI_Repackaged
#     wan2.2_animate_14B_bf16.safetensors (33 GB)
#     Use Comfy native UNETLoader with weight_dtype=fp8_e4m3fn at
#     runtime to fit in 24 GB VRAM (RTX 4090). RTX 5090 can use
#     default bf16.
#   - VAE: wan_2.1_vae.safetensors (Comfy-Org)
#   - Text encoder: umt5_xxl_fp8_e4m3fn_scaled.safetensors (Comfy-Org)
#   - LoRA (optional, recommended): wan2.2_animate_14B_relight_lora_bf16
#     (1.4 GB) — improves scene integration / relighting
mkdir -p "$MODELS/loras"
cat > "$ARIA2_INPUT" <<EOF
$HF/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_animate_14B_bf16.safetensors
  dir=$MODELS/diffusion_models
  out=wan2.2_animate_14B_bf16.safetensors

$HF/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors
  dir=$MODELS/vae
  out=wan_2.1_vae.safetensors

$HF/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
  dir=$MODELS/text_encoders
  out=umt5_xxl_fp8_e4m3fn_scaled.safetensors

$HF/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_animate_14B_relight_lora_bf16.safetensors
  dir=$MODELS/loras
  out=wan2.2_animate_14B_relight_lora_bf16.safetensors
EOF

ARIA_AUTH=()
[ -n "$TOKEN" ] && ARIA_AUTH=(--header="Authorization: Bearer $TOKEN")

# --auto-file-renaming=false + --allow-overwrite=true: resume partials,
# don't create .1 .2 duplicates. --max-tries=3: bail on persistent 404.
if aria2c -j 4 -x 16 -s 16 -k 1M --auto-file-renaming=false \
    --allow-overwrite=true --max-tries=3 --retry-wait=5 \
    "${ARIA_AUTH[@]}" -i "$ARIA2_INPUT" 2>&1 | tail -30 | tee -a "$LOG"; then
  log "  Model downloads OK"
else
  log "  WARN: one or more model downloads failed. Check URLs above and re-run."
  log "  Expected files (verify on HF if 404):"
  log "    Comfy-Org/Wan_2.2_ComfyUI_Repackaged/split_files/diffusion_models/wan2.2_animate_14B_bf16.safetensors"
  log "    Comfy-Org/Wan_2.2_ComfyUI_Repackaged/split_files/vae/wan_2.1_vae.safetensors"
  log "    Comfy-Org/Wan_2.2_ComfyUI_Repackaged/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
fi

# ─────────────────────────────────────────────
# 4. Fetch API code into /workspace/api/ and launch supervisors
# ─────────────────────────────────────────────
log "[4/4] Fetching API code and launching supervisors..."
mkdir -p /workspace/api
for f in main.py workflows.py safety.py setup.py; do
  wget -q "$API_REPO/$f" -O "/workspace/api/$f" && \
    log "  ✓ /workspace/api/$f" || \
    log "  ✗ /workspace/api/$f (skipped — file may not exist in repo yet)"
done

# Pull start scripts to /workspace
for f in start_comfy.sh start_api.sh; do
  wget -q "$API_REPO/$f" -O "/workspace/$f" && chmod +x "/workspace/$f" && \
    log "  ✓ /workspace/$f" || log "  ✗ /workspace/$f (skipped)"
done

# Write config.env for the supervisors to source
cat > /workspace/api/config.env <<EOF
PYTHON="$PYTHON"
COMFY_ROOT="$COMFY_ROOT"
API_DIR="/workspace/api"
API_REPO="$API_REPO"
HF_HOME="/workspace/hf-cache"
EOF
log "  Wrote /workspace/api/config.env"

# Stop status server before launching real services
if [ -f "$STATUS_PID_FILE" ]; then
  kill "$(cat "$STATUS_PID_FILE" 2>/dev/null)" 2>/dev/null || true
  rm -f "$STATUS_PID_FILE"
fi

# Launch ComfyUI supervisor (writes its own log + flock guard)
if [ -x "/workspace/start_comfy.sh" ]; then
  log "  Launching start_comfy.sh..."
  setsid nohup bash /workspace/start_comfy.sh </dev/null >/workspace/start_comfy.boot.log 2>&1 8>&- &
  sleep 1
fi

# Launch FastAPI supervisor
if [ -x "/workspace/start_api.sh" ]; then
  log "  Launching start_api.sh..."
  setsid nohup bash /workspace/start_api.sh </dev/null >/workspace/start_api.boot.log 2>&1 8>&- &
  sleep 1
fi

log "=========================================="
log "Setup complete. Tail logs:"
log "  /workspace/motion_setup.log    (this script)"
log "  /workspace/comfyui.log         (ComfyUI)"
log "  /workspace/api.log             (FastAPI)"
log "=========================================="
