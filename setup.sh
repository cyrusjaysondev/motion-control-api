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
  local url="$1"; local name="$2"; local pin="${3:-}"
  local dir="$NODES/$name"
  if [ -d "$dir/.git" ]; then
    log "  $name: pull..."
    (cd "$dir" && git pull --ff-only 2>&1 | tail -1)
  else
    log "  $name: clone..."
    # No --depth 1 — we may need to checkout an older SHA via the pin
    # arg, and unshallowing after a partial clone is slow + flaky.
    git clone "$url" "$dir" 2>&1 | tail -1
  fi
  if [ -n "$pin" ]; then
    log "  $name: pinning to $pin..."
    (cd "$dir" && git checkout "$pin" 2>&1 | tail -1)
  fi
  if [ -f "$dir/requirements.txt" ]; then
    log "  $name: pip install -r requirements.txt..."
    $PIP install -q -r "$dir/requirements.txt" 2>&1 | tail -1
  fi
}

# WanVideoWrapper pinned to d18cdb1 ("Fix offload on interrupt", May 5).
# The May 23 commit (5437b01, "Initial LongCatAvatar 1.5 support")
# introduced a code path that references an uninitialized variable
# `multitalk_audio_stride` in WanVideoSampler, breaking every job
# that doesn't pass the new multitalk audio inputs. Upstream issue
# not yet filed; pin to the last known-good before that landed.
install_node "https://github.com/kijai/ComfyUI-WanVideoWrapper" "ComfyUI-WanVideoWrapper" "d18cdb18597f525ef8d613a0cb447080fbab8fce"
install_node "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite" "ComfyUI-VideoHelperSuite"
install_node "https://github.com/Fannovel16/comfyui_controlnet_aux" "comfyui_controlnet_aux"
# AIGCTV/kijai pose-extraction nodes — VitPose + YOLO ONNX runtime,
# PoseAndFaceDetection, DrawViTPose. Required by workflows.py since
# the port to kijai's WanVideoWrapper graph.
install_node "https://github.com/kijai/ComfyUI-WanAnimatePreprocess" "ComfyUI-WanAnimatePreprocess"
# SAM2 segmentation for the use_sam2_mask + background_image path
# (character isolation + background swap on /motion). Provides
# Sam2Segmentation + DownloadAndLoadSAM2Model. The mask post-processing
# (GrowMaskWithBlur, BlockifyMask, DrawMaskOnImage) is already in
# ComfyUI-KJNodes installed by the base runpod/comfyui image.
install_node "https://github.com/kijai/ComfyUI-segment-anything-2" "ComfyUI-segment-anything-2"
log "  Done"

# ─────────────────────────────────────────────
# 3. Download Wan 2.2 Animate model weights (~30 GB)
#    Source: Kijai/WanVideo_comfy on HuggingFace — pre-packaged for
#    the ComfyUI-WanVideoWrapper node layout.
# ─────────────────────────────────────────────
log "[3/4] Downloading Wan 2.2 Animate models (AIGCTV/kijai stack)..."
mkdir -p "$MODELS/diffusion_models" "$MODELS/vae" "$MODELS/text_encoders" \
         "$MODELS/loras" "$MODELS/clip_vision" "$MODELS/detection" \
         "$MODELS/sam2"

ARIA2_INPUT="/tmp/motion-downloads.txt"
HF="https://huggingface.co"

# Build the aria2 input file. Each entry is:
#   <url>
#     dir=<target_directory>
#     out=<filename>
# `aria2c -i` reads this format. -x 16 -s 16 = 16 connections per file.
#
# Files (per the AIGCTV walkthrough — Wan 2.2 Animate via kijai's
# WanVideoWrapper graph, NOT Comfy's native WanAnimateToVideo path):
#
#   Diffusion: Wan2_2-Animate-14B_fp8_scaled_e4m3fn_KJ_v2 (17 GB)
#     Pre-quantized fp8 from Kijai/WanVideo_comfy_fp8_scaled.
#     Loaded by WanVideoModelLoader (not Comfy's UNETLoader).
#
#   VAE: Wan2_1_VAE_fp32 (500 MB) from Kijai/WanVideo_comfy.
#     fp32 is the "master" the wrapper expects; precision cast happens
#     in WanVideoVAELoader (we use bf16).
#
#   Text encoder: umt5-xxl-enc-fp8_e4m3fn (6.7 GB) from Kijai's repo.
#     Loaded by WanVideoTextEncodeCached (caches across runs).
#
#   CLIP Vision: clip_vision_h (1.2 GB) — identity encoding for the
#     character image so Wan doesn't drift in the first frames.
#
#   LoRAs:
#     - WanAnimate_relight_lora_fp16 (1.4 GB) — lighting consistency
#     - Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16 (~600 MB) —
#       4-step distillation, drops sampling from 20 → 4 steps. THE big
#       speed win in the AIGCTV stack.
#
#   Detection (kijai/ComfyUI-WanAnimatePreprocess):
#     - vitpose-l-wholebody.onnx (1.2 GB) — body + hands + face
#     - yolov10m.onnx (60 MB) — person bbox for cropping
cat > "$ARIA2_INPUT" <<EOF
$HF/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/Wan22Animate/Wan2_2-Animate-14B_fp8_scaled_e4m3fn_KJ_v2.safetensors
  dir=$MODELS/diffusion_models
  out=Wan2_2-Animate-14B_fp8_scaled_e4m3fn_KJ_v2.safetensors

$HF/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_fp32.safetensors
  dir=$MODELS/vae
  out=Wan2_1_VAE_fp32.safetensors

$HF/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-fp8_e4m3fn.safetensors
  dir=$MODELS/text_encoders
  out=umt5-xxl-enc-fp8_e4m3fn.safetensors

$HF/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors
  dir=$MODELS/clip_vision
  out=clip_vision_h.safetensors

$HF/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors
  dir=$MODELS/loras
  out=WanAnimate_relight_lora_fp16.safetensors

$HF/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22-Lightning/old/Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors
  dir=$MODELS/loras
  out=Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors

$HF/Kijai/WanVideo_comfy/resolve/main/FastWan/FastWan_T2V_14B_480p_lora_rank_128_bf16.safetensors
  dir=$MODELS/loras
  out=FastWan_T2V_14B_480p_lora_rank_128_bf16.safetensors

$HF/Kijai/WanVideo_comfy/resolve/main/Pusa/Wan22_PusaV1_lora_LOW_resized_dynamic_avg_rank_98_bf16.safetensors
  dir=$MODELS/loras
  out=Wan22_PusaV1_lora_LOW_resized_rank98_bf16.safetensors

$HF/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22_FunReward/Wan2.2-Fun-A14B-InP-LOW-HPS2.1_resized_dynamic_avg_rank_15_bf16.safetensors
  dir=$MODELS/loras
  out=Wan2.2-Fun-A14B-InP-LOW-HPS2.1_bf16.safetensors

$HF/Wan-AI/Wan2.2-Animate-14B/resolve/main/process_checkpoint/det/yolov10m.onnx
  dir=$MODELS/detection
  out=yolov10m.onnx

$HF/JunkyByte/easy_ViTPose/resolve/main/onnx/wholebody/vitpose-l-wholebody.onnx
  dir=$MODELS/detection
  out=vitpose-l-wholebody.onnx

$HF/Kijai/sam2-safetensors/resolve/main/sam2.1_hiera_base_plus.safetensors
  dir=$MODELS/sam2
  out=sam2.1_hiera_base_plus.safetensors
EOF

ARIA_AUTH=()
[ -n "$TOKEN" ] && ARIA_AUTH=(--header="Authorization: Bearer $TOKEN")

# --auto-file-renaming=false + --allow-overwrite=true: resume partials,
# don't create .1 .2 duplicates. --max-tries=3: bail on persistent 404.
if aria2c -j 4 -x 16 -s 16 -k 1M --auto-file-renaming=false \
    --allow-overwrite=false --max-tries=3 --retry-wait=5 \
    "${ARIA_AUTH[@]}" -i "$ARIA2_INPUT" 2>&1 | tail -30 | tee -a "$LOG"; then
  log "  Model downloads OK (~27 GB total — Wan KJ_v2 + VAE + umt5 + CLIP-H + 2 LoRAs + 2 ONNX detectors)"
else
  log "  WARN: one or more model downloads failed. Check URLs above and re-run."
  log "  Expected files (verify on HF if 404):"
  log "    Kijai/WanVideo_comfy_fp8_scaled/Wan22Animate/Wan2_2-Animate-14B_fp8_scaled_e4m3fn_KJ_v2.safetensors"
  log "    Kijai/WanVideo_comfy/Wan2_1_VAE_fp32.safetensors"
  log "    Kijai/WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors"
  log "    Comfy-Org/Wan_2.1_ComfyUI_repackaged/split_files/clip_vision/clip_vision_h.safetensors"
  log "    Kijai/WanVideo_comfy/LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors"
  log "    Kijai/WanVideo_comfy/LoRAs/Wan22-Lightning/old/Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors"
  log "    Wan-AI/Wan2.2-Animate-14B/process_checkpoint/det/yolov10m.onnx"
  log "    JunkyByte/easy_ViTPose/onnx/wholebody/vitpose-l-wholebody.onnx"
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
