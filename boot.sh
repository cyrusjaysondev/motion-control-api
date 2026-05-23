#!/bin/bash
# =============================================================
# motion-control-api — Container Start Command entrypoint
#
# Usage (RunPod template → Container Start Command):
#   bash -c "wget -qO /tmp/boot.sh https://raw.githubusercontent.com/cyrusjaysondev/motion-control-api/main/boot.sh && bash /tmp/boot.sh"
#
# RunPod's "Container Start Command" replaces the image's CMD.
# runpod/comfyui:latest ships /start.sh as CMD — that's what starts
# SSH, JupyterLab, FileBrowser, and ComfyUI's pre-baked instance. If
# you only invoke setup.sh, those services never come up. This script
# runs /start.sh in the background, fetches+runs setup.sh, then
# `wait`s on /start.sh so the container stays alive.
# =============================================================
set -u

log() { echo "[boot $(date '+%H:%M:%S')] $1"; }

SETUP_URL="https://raw.githubusercontent.com/cyrusjaysondev/motion-control-api/main/setup.sh"

log "running /start.sh in background (SSH / JupyterLab / ComfyUI)"
/start.sh &
START_PID=$!

# Give /start.sh a moment to bring DNS + venv online before we wget.
sleep 3

log "fetching setup.sh from $SETUP_URL"
if ! wget -qO /tmp/setup.sh "$SETUP_URL"; then
  log "ERROR: wget failed — check network. Container stays up so you can debug."
else
  if [ ! -s /tmp/setup.sh ]; then
    log "ERROR: downloaded setup.sh is empty — CDN or URL issue. Container stays up."
  else
    log "running setup.sh (first deploy: ~10-20 min for Wan model downloads)"
    bash /tmp/setup.sh || log "setup.sh exited non-zero (rc=$?) — container stays up for debugging"
  fi
fi

log "handing off to /start.sh (PID $START_PID) — container will stay alive"
wait "$START_PID"
