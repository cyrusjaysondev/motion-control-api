# motion-control-api

A dedicated FastAPI + ComfyUI service for **character motion transfer** using **Wan 2.2 Animate** (Alibaba, Sept 2025) — the state-of-the-art open-source model for taking a character image + a driving video and producing a video of the character performing the driving motion with identity preserved.

## Why this exists

Predecessor `/ltx/motion` in the `ai-gen-api-v2` service was built on LTX 2.3 + IC-LoRA Union-Control. After 42 iterations of tuning, we proved that the LTX 2.3 IC-LoRA was **never designed for image-driven character anchoring** — it's a prompt-driven motion-retargeting LoRA. Pushing it past its design intent produced (in order): identity drift, ghosting, mid-clip noise collapse, and per-frame face-swap flicker.

**Wan 2.2 Animate** is the architecturally correct fit: a unified animation + replacement model with spatially-aligned skeleton signals for body, implicit facial features for expressions, and a relighting LoRA for scene integration. Identity preservation is baked in.

## Reference

- Paper: [arxiv 2509.14055](https://arxiv.org/html/2509.14055v1)
- Model repo: [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)
- ComfyUI integration: [kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)
- Project page: [humanaigc.github.io/wan-animate](https://humanaigc.github.io/wan-animate/)

## Service architecture

```
RunPod pod (RTX 5090, 32 GB VRAM)
├── ComfyUI on :8188 (with WanVideoWrapper + controlnet_aux)
└── FastAPI on :7860 (this service)

Network volume: motion-models-eu-ro-1 (50 GB)
└── /workspace/runpod-slim/ComfyUI/models/
    ├── diffusion_models/wan22_animate_14b_fp16.safetensors  (~28 GB)
    ├── text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors (~6 GB)
    └── vae/wan_2.1_vae.safetensors                           (~250 MB)
```

## API surface

```
POST /motion
  reference_video: file  (driving motion, mp4/mov/webm)
  image:           file  (character to animate)
  prompt:          str   (optional, describes the scene)
  length:          int   (default 81, output frames)
  width / height:  int   (default 720×1280)
  fps:             int   (default 16, Wan native rate)
  seed:            int   (default -1)
  audio:           bool  (mux reference audio onto output)
  → { job_id, poll_url }

GET  /status/{job_id}
GET  /jobs
DELETE /jobs/{id}/cancel
GET  /video/{filename}
GET  /image/{filename}
GET  /healthz

POST /admin/install-comfy-node
POST /admin/restart-comfyui
GET  /admin/comfy-status
GET  /admin/comfy-objects
```

## Deploy

1. Provision RunPod pod from the `motion-control-api` template
2. Pod's `dockerStartCmd` wgets `boot.sh` from this repo
3. `boot.sh` runs `/start.sh` (SSH/Jupyter) + fetches `setup.sh`
4. `setup.sh` installs ComfyUI custom nodes, downloads Wan model weights to the network volume, launches `start_comfy.sh` + `start_api.sh`
5. After ~5-15 min (first deploy downloads ~30 GB of model weights), the API is live

## Known limitations (per upstream research)

- Identity drift after ~6 s in a single take (mitigation: segment into 4–5 s chunks with locked seed)
- Color/exposure creep over the clip (mitigation: V2V enhancer second pass)
- Hands and complex multi-object interactions are the weakest area
- Frame budget ~240 frames on 24 GB VRAM at 768×1344 (5090 with 32 GB removes the ceiling)

## Relationship to `ai-gen-api-v2`

This is an **independent service**. The `ai-gen-api-v2` `/ltx/motion` endpoint is kept as-is for backwards compatibility but is superseded by this one for any new client integrations. The two services share NO state and run on separate pods, separate network volumes, separate GitHub repos.
