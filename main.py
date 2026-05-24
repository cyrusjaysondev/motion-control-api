"""motion-control-api — FastAPI service.

Single-purpose: take a reference video + a character image, return a
video of the character performing the reference motion using Wan 2.2
Animate. Backend is ComfyUI on the same pod (:8188).

Pipeline patterns lifted from ai-gen-api-v2:
  - run_job        — background ComfyUI job runner + WebSocket wait
  - _mux_reference_audio — ffmpeg mux reference audio onto silent output
  - /admin/*       — install-comfy-node, restart-comfyui, comfy-status
  - setup.py shim  — code-refresh via pip-install side effect

The Wan workflow itself (build_wan_motion_workflow) lives in workflows.py
and is filled in during Phase 6 of the deploy plan. This file is the
scaffolding around it: file uploads, ffmpeg normalize, job lifecycle.
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
import websockets
from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Workflow builders. Phase 6 implements build_wan_motion_workflow().
try:
    from workflows import build_wan_motion_workflow
except ImportError:
    build_wan_motion_workflow = None  # type: ignore  # scaffolding-time placeholder

app = FastAPI(title="motion-control-api", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ─── Paths + config ───────────────────────────────────────────
COMFY_ROOT = os.environ.get("COMFY_ROOT", "/workspace/runpod-slim/ComfyUI")
INPUT_DIR = Path(COMFY_ROOT) / "input"
OUTPUT_DIR = Path(COMFY_ROOT) / "output"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMFYUI_URL = "http://127.0.0.1:8188"
POD_ID = os.environ.get("RUNPOD_POD_ID", "local")
BASE_URL = f"https://{POD_ID}-7860.proxy.runpod.net" if POD_ID != "local" else "http://localhost:7860"

_VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".gif"}

# In-memory job state. For a single-pod service this is fine; if we
# later run multiple pods behind a load balancer, swap for Redis.
jobs: dict[str, dict] = {}


# ─── Helpers (lifted from ai-gen-api-v2, kept here so this service
# has no external dependency on that repo) ────────────────────

def _mux_reference_audio(video_path: Path, audio_source: Path) -> tuple[bool, str]:
    """Replace audio on `video_path` with audio from `audio_source`.
    Loops short refs (-stream_loop -1), trims to video duration
    (-shortest). No-op if the source has no audio stream.
    """
    if not audio_source.exists():
        return False, "audio source missing"
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0",
             str(audio_source)],
            capture_output=True, timeout=15,
        )
    except Exception as e:
        return False, f"ffprobe error: {e}"
    if probe.returncode != 0 or b"audio" not in probe.stdout:
        return False, "reference has no audio track"

    tmp_out = video_path.with_name(f"{video_path.stem}.tmp{video_path.suffix}")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-stream_loop", "-1", "-i", str(audio_source),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(tmp_out),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=180)
    except Exception as e:
        tmp_out.unlink(missing_ok=True)
        return False, f"ffmpeg error: {e}"
    if res.returncode != 0 or not tmp_out.exists() or tmp_out.stat().st_size == 0:
        tmp_out.unlink(missing_ok=True)
        return False, f"ffmpeg failed: {(res.stderr or b'').decode(errors='replace')[-300:]}"
    os.replace(str(tmp_out), str(video_path))
    return True, "ok"


def _extract_video_thumbnail(video_path: Path) -> Path | None:
    thumb = video_path.with_name(f"{video_path.stem}_thumb.jpg")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path),
             "-vframes", "1", "-vf", "scale=320:-2", str(thumb)],
            capture_output=True, timeout=30,
        )
        return thumb if thumb.exists() and thumb.stat().st_size > 0 else None
    except Exception:
        return None


# ─── Core job runner ──────────────────────────────────────────

async def run_job(job_id: str, workflow: dict,
                  cleanup_paths: list | None = None,
                  audio_source_path: str | None = None):
    jobs[job_id] = {**jobs.get(job_id, {}),
                    "status": "processing",
                    "started_at": datetime.now(timezone.utc).isoformat()}
    try:
        client_id = str(uuid.uuid4())
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{COMFYUI_URL}/prompt",
                json={"prompt": workflow, "client_id": client_id},
            )
            if resp.status_code != 200:
                jobs[job_id] = {**jobs[job_id], "status": "failed", "error": resp.text}
                return
            prompt_id = resp.json()["prompt_id"]

        # Watch ComfyUI for completion. Two signals — whichever lands
        # first wins:
        #   (a) WebSocket "executing" frame with node=null + matching
        #       prompt_id → the canonical "execution finished" event.
        #   (b) Periodic /history poll → in case ComfyUI finished
        #       before the WS connection was established (small race
        #       window we hit on short, cached prompts), or the WS
        #       dropped silently. The poll runs every 3 s and breaks
        #       out the moment the prompt_id appears in history.
        ws_url = f"ws://127.0.0.1:8188/ws?clientId={client_id}"

        async def _ws_wait():
            try:
                async with websockets.connect(
                    ws_url, ping_interval=None, close_timeout=None, max_size=None,
                ) as ws:
                    while True:
                        raw = await ws.recv()
                        if isinstance(raw, bytes):
                            continue
                        msg = json.loads(raw)
                        if msg.get("type") == "executing":
                            data = msg.get("data", {})
                            if data.get("node") is None and data.get("prompt_id") == prompt_id:
                                return
            except Exception as e:
                print(f"[{job_id}] WS error: {e}", flush=True)
                # Fall through to history polling below — it'll detect
                # completion if/when it happens.
                return

        async def _history_poll():
            async with httpx.AsyncClient() as poll_client:
                while True:
                    await asyncio.sleep(3)
                    try:
                        r = await poll_client.get(f"{COMFYUI_URL}/history/{prompt_id}")
                        if r.status_code == 200 and r.json().get(prompt_id):
                            return
                    except Exception:
                        # transient; keep polling
                        pass

        ws_task = asyncio.create_task(_ws_wait())
        poll_task = asyncio.create_task(_history_poll())
        done, pending = await asyncio.wait(
            [ws_task, poll_task],
            return_when=asyncio.FIRST_COMPLETED,
            timeout=1800,  # 30-min hard ceiling
        )
        for t in pending:
            t.cancel()

        async with httpx.AsyncClient() as client:
            history = await client.get(f"{COMFYUI_URL}/history/{prompt_id}")
            job_data = history.json().get(prompt_id, {})
            status = job_data.get("status", {}).get("status_str", "")
            if status == "error":
                for m in job_data.get("status", {}).get("messages", []):
                    if m[0] == "execution_error":
                        jobs[job_id] = {**jobs[job_id], "status": "failed",
                                        "error": m[1].get("exception_message")}
                        return
            outputs = job_data.get("outputs", {})

        for node_output in outputs.values():
            for key in ("videos", "gifs", "images"):
                if key in node_output:
                    item = node_output[key][0]
                    filename = item["filename"]
                    subfolder = item.get("subfolder", "")
                    path = OUTPUT_DIR / subfolder / filename if subfolder else OUTPUT_DIR / filename
                    if path.exists():
                        ext = Path(filename).suffix.lower()
                        url = f"{BASE_URL}/{'image' if ext in ('.png','.jpg','.jpeg','.webp') else 'video'}/{filename}"

                        # Reference-audio mux (only when caller passed audio_source_path)
                        audio_warning = None
                        if audio_source_path and ext in _VIDEO_EXTS:
                            try:
                                ok, msg = await asyncio.to_thread(
                                    _mux_reference_audio, path, Path(audio_source_path),
                                )
                                if not ok:
                                    audio_warning = msg
                                print(f"[{job_id}] audio mux: {msg}")
                            except Exception as e:
                                audio_warning = str(e)
                                print(f"[{job_id}] audio mux raised: {e}")

                        thumbnail_url = None
                        if ext in _VIDEO_EXTS:
                            thumb = _extract_video_thumbnail(path)
                            if thumb is not None:
                                thumbnail_url = f"{BASE_URL}/image/{thumb.name}"

                        completed_at = datetime.now(timezone.utc)
                        duration_seconds = None
                        created_at = jobs[job_id].get("created_at")
                        if created_at:
                            started = datetime.fromisoformat(created_at)
                            duration_seconds = round((completed_at - started).total_seconds(), 1)

                        completed = {
                            "status": "completed",
                            "url": url,
                            "filename": filename,
                            "completed_at": completed_at.isoformat(),
                            "duration_seconds": duration_seconds,
                        }
                        if thumbnail_url:
                            completed["thumbnail_url"] = thumbnail_url
                        if audio_warning:
                            completed["audio_warning"] = audio_warning
                        jobs[job_id] = completed
                        return

        jobs[job_id] = {**jobs[job_id], "status": "failed", "error": "No output found"}
    except Exception as e:
        import traceback as _tb
        tb_str = _tb.format_exc()
        print(f"[{job_id}] run_job exception:\n{tb_str}", flush=True)
        jobs[job_id] = {
            **jobs[job_id],
            "status": "failed",
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": tb_str.splitlines()[-15:],
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
    finally:
        if cleanup_paths:
            for p in cleanup_paths:
                Path(p).unlink(missing_ok=True)


# ─── Health + job-management endpoints ────────────────────────

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": "motion-control-api", "version": app.version}


# /health alias so the API Explorer's "System → Health Check" tile
# (which calls GET /health, ai-gen-api-v2 convention) works against
# the motion-control pod without a per-endpoint URL override.
@app.get("/health")
async def health():
    return {"status": "ok", "service": "motion-control-api", "version": app.version}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/jobs")
async def get_all_jobs():
    return {
        "total": len(jobs),
        "summary": {s: sum(1 for j in jobs.values() if j.get("status") == s)
                    for s in ("queued", "processing", "completed", "failed")},
        "jobs": [{"job_id": jid, **info} for jid, info in jobs.items()],
    }


@app.get("/queue")
async def get_queue():
    """Active queue — anything our API thinks is queued/processing,
    PLUS anything ComfyUI itself is currently running or has pending.
    The ComfyUI-side merge is the recovery path: when uvicorn restarts
    (any code deploy), the in-memory `jobs` dict is wiped, but the
    workflow ComfyUI is sampling keeps running and stays visible here.
    Those orphaned jobs show with `source: "comfyui"` and the ComfyUI
    prompt_id as the job_id — they won't appear in /jobs or
    /status/{id} (those still need our API to know about the job)."""
    active = {jid: {"job_id": jid, "status": info.get("status"), "source": "api"}
              for jid, info in jobs.items()
              if info.get("status") in ("queued", "processing")}

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{COMFYUI_URL}/queue")
            if r.status_code == 200:
                comfy_q = r.json()
                # queue_running entries look like: [priority, prompt_id, prompt, extra, outputs]
                for entry in comfy_q.get("queue_running", []) or []:
                    pid = entry[1] if len(entry) > 1 else None
                    if pid and pid not in active:
                        active[pid] = {"job_id": pid, "status": "processing",
                                       "source": "comfyui"}
                for entry in comfy_q.get("queue_pending", []) or []:
                    pid = entry[1] if len(entry) > 1 else None
                    if pid and pid not in active:
                        active[pid] = {"job_id": pid, "status": "queued",
                                       "source": "comfyui"}
    except Exception as e:
        # Don't fail the whole call if ComfyUI is unreachable —
        # the API's own jobs dict is still useful.
        return {
            "count": len(active),
            "jobs": list(active.values()),
            "comfyui_unreachable": str(e),
        }

    return {"count": len(active), "jobs": list(active.values())}


@app.get("/videos")
async def list_videos():
    """List output videos on disk. motion-control-api writes flat
    into ComfyUI's output dir (no video/ + images/ split), so we
    glob OUTPUT_DIR directly for *.mp4 and attach the matching
    `<stem>_thumb.jpg` if present."""
    if not OUTPUT_DIR.exists():
        return {"total": 0, "videos": []}
    videos = []
    for f in sorted(OUTPUT_DIR.glob("*.mp4"),
                    key=lambda x: x.stat().st_mtime, reverse=True):
        stat = f.stat()
        entry = {
            "filename": f.name,
            "size_mb": round(stat.st_size / 1024 / 1024, 2),
            "url": f"{BASE_URL}/video/{f.name}",
            "created_at": stat.st_mtime,
        }
        thumb = OUTPUT_DIR / f"{f.stem}_thumb.jpg"
        if thumb.exists():
            entry["thumbnail_url"] = f"{BASE_URL}/image/{thumb.name}"
        videos.append(entry)
    return {"total": len(videos), "videos": videos}


@app.delete("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job.get("status") in ("completed", "failed"):
        raise HTTPException(400, f"Job already {job.get('status')}")
    try:
        async with httpx.AsyncClient() as client:
            await client.post(f"{COMFYUI_URL}/queue", json={"delete": [job_id]})
    except Exception:
        pass
    jobs[job_id] = {"status": "cancelled"}
    return {"job_id": job_id, "status": "cancelled"}


@app.get("/video/{filename}")
async def serve_video(filename: str):
    # Sanitize: filename only, no traversal
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(400, "invalid filename")
    for sub in ("", "video"):
        path = OUTPUT_DIR / sub / filename if sub else OUTPUT_DIR / filename
        if path.exists():
            return FileResponse(str(path))
    raise HTTPException(404, "Video not found")


@app.get("/image/{filename}")
async def serve_image(filename: str):
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(400, "invalid filename")
    for sub in ("", "images", "video"):  # thumbnails may live under video/
        path = OUTPUT_DIR / sub / filename if sub else OUTPUT_DIR / filename
        if path.exists():
            return FileResponse(str(path))
    raise HTTPException(404, "Image not found")


# ─── /motion — the only generation endpoint ───────────────────

@app.post("/motion")
async def motion(
    background_tasks: BackgroundTasks,
    reference_video: UploadFile = File(..., description="Driving motion video (any common format)."),
    image: UploadFile = File(..., description="Character image — identity source."),
    prompt: str = Form("", description="Optional scene description."),
    negative_prompt: str = Form("low quality, blurry, distorted, deformed, motion artifacts",
                                description="What to avoid."),
    length: int = Form(81, description="Output frame count. Wan native is 16fps so 81 ≈ 5s.",
                       ge=17, le=240),
    width: int = Form(720, description="Output width.", ge=256, le=1280),
    height: int = Form(1280, description="Output height.", ge=256, le=1920),
    fps: int = Form(16, description="Output fps. Wan native is 16; we re-time in ffmpeg if user wants 24/30."),
    seed: int = Form(-1),
    audio: bool = Form(True, description="Carry the reference video's audio onto the output."),
    lightning_steps: int = Form(4, description="Sampler steps. Lightning LoRA is calibrated for 4; bump to 6-8 for marginal quality gains at proportional cost.",
                                ge=2, le=20),
    sampler_shift: float = Form(5.0, description="Noise schedule shift. 5.0 fits Lightning; 3.0 works for non-distilled, 7-8 for high-res (≥1024p).",
                                ge=1.0, le=10.0),
    relight: bool = Form(True, description="Apply the WanAnimate relight LoRA for lighting consistency between character and scene."),
):
    """Wan 2.2 Animate character motion transfer.

    Pipeline:
      1. Save uploaded files to ComfyUI input dir
      2. ffmpeg normalize reference video (fps + canvas)
      3. Build the Wan workflow (workflows.build_wan_motion_workflow)
      4. Submit to ComfyUI, poll via WebSocket
      5. ffmpeg-mux reference audio onto the silent output
      6. Return job_id + poll URL
    """
    if build_wan_motion_workflow is None:
        raise HTTPException(503,
            "workflows.build_wan_motion_workflow not available — Phase 6 implementation pending.")

    # Save character image
    img_ext = (image.filename or "").lower().rsplit(".", 1)[-1] or "png"
    if img_ext not in ("png", "jpg", "jpeg", "webp"):
        img_ext = "png"
    img_filename = f"motion_img_{uuid.uuid4().hex}.{img_ext}"
    img_path = str(INPUT_DIR / img_filename)
    Path(img_path).write_bytes(await image.read())

    # Save raw reference video
    raw_video_bytes = await reference_video.read()
    REF_MAX_BYTES = 100 * 1024 * 1024
    if len(raw_video_bytes) > REF_MAX_BYTES:
        Path(img_path).unlink(missing_ok=True)
        raise HTTPException(413, f"reference video too large: {len(raw_video_bytes)//(1024*1024)} MB > 100 MB.")
    raw_video_ext = (reference_video.filename or "").lower().rsplit(".", 1)[-1] or "mp4"
    raw_video_path = str(INPUT_DIR / f"motion_ref_raw_{uuid.uuid4().hex}.{raw_video_ext}")
    Path(raw_video_path).write_bytes(raw_video_bytes)

    ref_video_filename = f"motion_ref_{uuid.uuid4().hex}.mp4"
    ref_video_path = str(INPUT_DIR / ref_video_filename)

    # ffmpeg normalize: target fps + canvas, strip audio (we re-mux later)
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-stream_loop", "-1", "-i", raw_video_path,
        "-vf", (
            f"scale=w={width}:h={height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"fps={fps}"
        ),
        "-frames:v", str(length),
        "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
        ref_video_path,
    ]

    def _run_ffmpeg() -> subprocess.CompletedProcess:
        return subprocess.run(ffmpeg_cmd, check=True, capture_output=True, timeout=120)

    try:
        await asyncio.to_thread(_run_ffmpeg)
    except subprocess.CalledProcessError as e:
        Path(raw_video_path).unlink(missing_ok=True)
        Path(img_path).unlink(missing_ok=True)
        raise HTTPException(400,
            f"could not decode reference video: {(e.stderr or b'').decode(errors='replace')[:500]}")
    except subprocess.TimeoutExpired:
        Path(raw_video_path).unlink(missing_ok=True)
        Path(img_path).unlink(missing_ok=True)
        raise HTTPException(408, "reference video normalization timed out (>120s)")

    # Hold the raw video around for audio mux (if requested), else clean it up now
    cleanup_paths: list[str] = [img_path, ref_video_path]
    audio_source_path = raw_video_path if audio else None
    if audio:
        cleanup_paths.append(raw_video_path)
    else:
        Path(raw_video_path).unlink(missing_ok=True)

    workflow = build_wan_motion_workflow(
        reference_video_filename=ref_video_filename,
        character_image_filename=img_filename,
        prompt=prompt, negative_prompt=negative_prompt,
        width=width, height=height, length=length, fps=fps, seed=seed,
        relight=relight,
        lightning_steps=lightning_steps,
        sampler_shift=sampler_shift,
    )

    # Pull the auto-sized swap + tile values out for the response so
    # callers can see what the workload-scaler chose. The block_swap
    # node is "103" in workflows.build_wan_motion_workflow.
    chosen_blocks_to_swap = workflow.get("103", {}).get("inputs", {}).get("blocks_to_swap")
    chosen_tile_vae = workflow.get("200", {}).get("inputs", {}).get("enable_vae_tiling")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
    background_tasks.add_task(run_job, job_id, workflow, cleanup_paths, audio_source_path)
    return {
        "job_id": job_id,
        "status": "queued",
        "model": "wan-2.2-animate-14b",
        "poll_url": f"{BASE_URL}/status/{job_id}",
        "ref_video_normalized_to": {"width": width, "height": height, "fps": fps, "frames": length},
        "audio_source": "reference" if audio else "none",
        "vram_config": {
            "blocks_to_swap": chosen_blocks_to_swap,
            "vae_tiling": chosen_tile_vae,
            "megapixel_frames": round(width * height * (2 * length) / 1_000_000, 1),
        },
    }


# ─── Admin endpoints (proven from ai-gen-api-v2) ──────────────

def _require_admin(authorization: str | None) -> None:
    token = os.environ.get("ADMIN_TOKEN")
    if not token:
        return  # admin endpoints open when ADMIN_TOKEN is unset
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing Bearer token")
    if authorization.removeprefix("Bearer ").strip() != token:
        raise HTTPException(401, "invalid token")


def _detect_comfy_root() -> str:
    for candidate in ("/workspace/runpod-slim/ComfyUI", "/workspace/ComfyUI", COMFY_ROOT):
        if (Path(candidate) / "main.py").is_file():
            return candidate
    raise HTTPException(500, "ComfyUI root not found")


@app.get("/admin/comfy-status")
async def admin_comfy_status(authorization: str = Header(default=None)):
    _require_admin(authorization)
    comfy_root = _detect_comfy_root()
    nodes_dir = Path(comfy_root) / "custom_nodes"
    on_disk = []
    if nodes_dir.is_dir():
        for entry in sorted(nodes_dir.iterdir()):
            if entry.is_dir():
                on_disk.append({
                    "name": entry.name,
                    "has_init": (entry / "__init__.py").is_file(),
                    "has_requirements": (entry / "requirements.txt").is_file(),
                    "has_git": (entry / ".git").is_dir(),
                })

    loaded = 0
    key_loaded: dict[str, bool] = {}
    object_info_error = None
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(f"{COMFYUI_URL}/object_info")
            if r.status_code == 200:
                data = r.json()
                loaded = len(data)
                for n in ("WanVideoModelLoader", "WanVideoSampler",
                          "WanVideoDecode", "WanVideoTextEncode",
                          "VHS_LoadVideo", "VHS_VideoCombine",
                          "DWPreprocessor"):
                    key_loaded[n] = n in data
            else:
                object_info_error = f"HTTP {r.status_code}"
    except Exception as e:
        object_info_error = str(e)

    shim_log: list[str] = []
    log_path = Path("/workspace/motion-shim.log")
    if log_path.exists():
        try:
            shim_log = log_path.read_text(errors="replace").splitlines()[-50:]
        except Exception:
            pass

    return {
        "comfy_root": comfy_root,
        "comfy_object_info_error": object_info_error,
        "custom_nodes_on_disk": on_disk,
        "loaded_node_count": loaded,
        "key_nodes_loaded": key_loaded,
        "shim_log_tail": shim_log,
    }


@app.get("/admin/comfy-objects")
async def admin_comfy_objects(filter: str = "", authorization: str = Header(default=None)):
    _require_admin(authorization)
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(f"{COMFYUI_URL}/object_info")
            if r.status_code != 200:
                raise HTTPException(502, f"ComfyUI returned HTTP {r.status_code}")
            data = r.json()
    except httpx.HTTPError as e:
        raise HTTPException(502, f"ComfyUI unreachable: {e}")
    names = sorted(k for k in data if filter.lower() in k.lower()) if filter else sorted(data.keys())
    return {"count": len(names), "filter": filter, "names": names}


@app.post("/admin/install-comfy-node")
async def admin_install_comfy_node(
    repo: str = Form(..., description="github slug 'owner/name' OR full https:// URL."),
    restart_comfyui: bool = Form(True),
    authorization: str = Header(default=None),
):
    _require_admin(authorization)
    repo_slug = repo.strip()
    if not repo_slug:
        raise HTTPException(400, "repo is required")
    if "://" not in repo_slug:
        repo_slug = f"https://github.com/{repo_slug}"
    repo_name = repo_slug.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")

    comfy_root = _detect_comfy_root()
    nodes_dir = Path(comfy_root) / "custom_nodes"
    nodes_dir.mkdir(parents=True, exist_ok=True)
    target_dir = nodes_dir / repo_name
    trace: list[str] = []

    def _run(cmd: list[str], cwd: str | None = None) -> tuple[int, str]:
        try:
            res = subprocess.run(cmd, cwd=cwd, capture_output=True, timeout=240)
            output = (res.stdout + res.stderr).decode(errors="replace")
            trace.append(f"$ {' '.join(cmd)}\n{output[-1000:]}\n--- exit {res.returncode} ---")
            return res.returncode, output
        except subprocess.TimeoutExpired:
            trace.append(f"$ {' '.join(cmd)}\n  TIMEOUT (>240s)")
            return 124, ""

    if not (target_dir / ".git").is_dir():
        if target_dir.exists():
            trace.append(f"removing corrupt {target_dir}")
            subprocess.run(["rm", "-rf", str(target_dir)], check=False)
        rc, _ = _run(["git", "clone", repo_slug, str(target_dir)])
        if rc != 0:
            raise HTTPException(500, f"git clone failed: {trace[-1] if trace else ''}")
    else:
        _run(["git", "pull", "--ff-only"], cwd=str(target_dir))

    req = target_dir / "requirements.txt"
    if req.is_file():
        _run([sys.executable, "-m", "pip", "install", "-r", str(req)])
    else:
        trace.append("(no requirements.txt)")

    restarted = False
    if restart_comfyui:
        try:
            ns = subprocess.run(["netstat", "-tlnp"], capture_output=True, timeout=10)
            comfy_pid: str | None = None
            for line in (ns.stdout or b"").decode(errors="replace").splitlines():
                if ":8188" not in line:
                    continue
                tail = line.split()[-1] if line.split() else ""
                if "/" in tail:
                    pid_part = tail.split("/")[0]
                    if pid_part.isdigit():
                        comfy_pid = pid_part
                        break
            if comfy_pid:
                rc, _ = _run(["kill", "-9", comfy_pid])
                trace.append(f"kill -9 {comfy_pid} (ComfyUI :8188 owner) → exit {rc}")
                if rc == 0:
                    restarted = True
            else:
                trace.append("netstat -tlnp found no :8188 listener")
        except Exception as e:
            trace.append(f"netstat-based kill failed: {e}")
        if not restarted:
            rc, _ = _run(["pkill", "-9", "-f", "main.py --listen 0.0.0.0 --port 8188"])
            trace.append(f"pkill fallback → exit {rc}")
            if rc == 0:
                restarted = True

    return {
        "ok": True,
        "repo": repo_slug,
        "target_dir": str(target_dir),
        "has_init_py": (target_dir / "__init__.py").is_file(),
        "comfyui_restart_signaled": restarted,
        "trace": trace,
    }


@app.post("/admin/restart-comfyui")
async def admin_restart_comfyui(authorization: str = Header(default=None)):
    _require_admin(authorization)
    trace: list[str] = []
    matched: str | None = None
    killed_pid: str | None = None
    try:
        netstat = subprocess.run(["netstat", "-tlnp"], capture_output=True, timeout=10)
        for line in (netstat.stdout or b"").decode(errors="replace").splitlines():
            if ":8188" not in line:
                continue
            tail = line.split()[-1] if line.split() else ""
            if "/" in tail:
                pid_part = tail.split("/")[0]
                if pid_part.isdigit():
                    killed_pid = pid_part
                    break
        if killed_pid:
            res = subprocess.run(["kill", "-9", killed_pid], capture_output=True, timeout=10)
            trace.append(f"kill -9 {killed_pid} (port :8188 owner) → exit {res.returncode}")
            if res.returncode == 0:
                matched = f"port:8188:pid={killed_pid}"
        else:
            trace.append("netstat -tlnp showed no owner for :8188")
    except Exception as e:
        trace.append(f"netstat path error: {e}")

    if matched is None:
        try:
            res = subprocess.run(
                ["pkill", "-9", "-f", "main.py --listen 0.0.0.0 --port 8188"],
                capture_output=True, timeout=10,
            )
            trace.append(f"pkill fallback → exit {res.returncode}")
            if res.returncode == 0:
                matched = "pattern:start_comfy_argv"
        except Exception as e:
            trace.append(f"pkill fallback error: {e}")

    return {
        "ok": matched is not None,
        "matched_pattern": matched,
        "note": "Wait ~30s and poll /admin/comfy-status to verify the new node count.",
        "trace": trace,
    }


@app.get("/admin/log")
async def admin_log(
    name: str = "api",
    lines: int = 200,
    authorization: str = Header(default=None),
):
    """Tail one of the on-disk logs so we can read tracebacks without SSH.
    name=api → /workspace/api.log
    name=comfy → /workspace/comfy.log (whatever start_comfy.sh writes)
    name=setup → /workspace/motion_setup.log
    name=shim → /workspace/motion-shim.log"""
    _require_admin(authorization)
    log_map = {
        "api": "/workspace/api.log",
        "api_setup": "/workspace/api_setup.log",
        "comfy": "/workspace/comfy.log",
        "comfy_setup": "/workspace/comfy_setup.log",
        "setup": "/workspace/motion_setup.log",
        "shim": "/workspace/motion-shim.log",
    }
    path = log_map.get(name)
    if not path:
        raise HTTPException(400, f"unknown log '{name}' — known: {list(log_map.keys())}")
    if not Path(path).is_file():
        raise HTTPException(404, f"{path} does not exist")
    try:
        res = subprocess.run(
            ["tail", "-n", str(max(1, min(lines, 2000))), path],
            capture_output=True, timeout=10,
        )
        return {
            "path": path,
            "lines_requested": lines,
            "body": (res.stdout + res.stderr).decode(errors="replace"),
        }
    except Exception as e:
        raise HTTPException(500, f"tail failed: {e}")


@app.get("/admin/disk-status")
async def admin_disk_status(
    log_name: str = "",
    log_lines: int = 100,
    authorization: str = Header(default=None),
):
    """Run df + a couple of du checks so we can see what the kernel
    thinks the filesystem looks like — used when /motion fails with
    'Disk quota exceeded (os error 122)' and we can't SSH in.

    Optional log_name= reads a tail of /workspace/{log_name}.log so
    you can also pull tracebacks. Supported names: api, api_setup,
    comfy, comfy_setup, setup (= motion_setup), shim (= motion-shim)."""
    _require_admin(authorization)
    results: dict = {}
    try:
        res = subprocess.run(["df", "-h"], capture_output=True, timeout=10)
        results["df_h"] = (res.stdout + res.stderr).decode(errors="replace")
    except Exception as e:
        results["df_h_error"] = str(e)
    try:
        res = subprocess.run(["df", "-i"], capture_output=True, timeout=10)
        results["df_i"] = (res.stdout + res.stderr).decode(errors="replace")
    except Exception as e:
        results["df_i_error"] = str(e)
    for path in ("/workspace", "/workspace/api", str(INPUT_DIR), str(OUTPUT_DIR), "/tmp"):
        try:
            res = subprocess.run(["du", "-sh", path], capture_output=True, timeout=30)
            results[f"du_{path}"] = (res.stdout + res.stderr).decode(errors="replace").strip()
        except Exception as e:
            results[f"du_{path}_error"] = str(e)
    # Largest entries under /workspace
    try:
        res = subprocess.run(
            ["du", "-sh", "/workspace/runpod-slim", "/workspace/api",
             "/workspace/runpod-slim/ComfyUI/models",
             "/workspace/runpod-slim/ComfyUI/input",
             "/workspace/runpod-slim/ComfyUI/output"],
            capture_output=True, timeout=60,
        )
        results["workspace_breakdown"] = (res.stdout + res.stderr).decode(errors="replace").strip()
    except Exception as e:
        results["workspace_breakdown_error"] = str(e)
    # Inline log tail (so we don't need a separate /admin/log endpoint
    # before the GitHub raw CDN catches up).
    if log_name:
        log_map = {
            "api": "/workspace/api.log",
            "api_setup": "/workspace/api_setup.log",
            "comfy": "/workspace/comfy.log",
            "comfy_setup": "/workspace/comfy_setup.log",
            "setup": "/workspace/motion_setup.log",
            "shim": "/workspace/motion-shim.log",
        }
        log_path = log_map.get(log_name)
        if log_path and Path(log_path).is_file():
            try:
                res = subprocess.run(
                    ["tail", "-n", str(max(1, min(log_lines, 2000))), log_path],
                    capture_output=True, timeout=10,
                )
                results["log_path"] = log_path
                results["log_body"] = (res.stdout + res.stderr).decode(errors="replace")
            except Exception as e:
                results["log_error"] = str(e)
        else:
            results["log_path"] = log_path
            results["log_error"] = "log file not found or unknown name"
    return results


@app.post("/admin/exec-shell")
async def admin_exec_shell(
    cmd: str = Form(..., description="single shell command to run (no & background, no chained ;)"),
    timeout: int = Form(30),
    authorization: str = Header(default=None),
):
    """Run a single shell command as the API process user. Used when we
    need to diagnose or remount the volume without SSH. The command is
    passed to /bin/sh -c so & redirections work, but we still cap the
    timeout and return stdout+stderr+exit. Output is truncated at 16 KB.

    Examples of commands we expect to need:
      df -h /workspace
      mount | grep workspace
      mount -o remount /workspace
      umount /workspace && mount -t mfsmount mfsmaster ...
    """
    _require_admin(authorization)
    if not cmd.strip():
        raise HTTPException(400, "cmd is empty")
    try:
        res = subprocess.run(
            ["/bin/sh", "-c", cmd],
            capture_output=True, timeout=max(1, min(timeout, 120)),
        )
        out = (res.stdout + res.stderr).decode(errors="replace")
        return {
            "cmd": cmd,
            "exit_code": res.returncode,
            "output": out[-16384:],
            "truncated": len(out) > 16384,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "cmd": cmd,
            "exit_code": 124,
            "error": f"timeout after {timeout}s",
            "stdout_partial": (e.stdout or b"").decode(errors="replace")[-4096:],
        }


@app.post("/admin/restart-api")
async def admin_restart_api(authorization: str = Header(default=None)):
    """Self-terminate so start_api.sh's while-loop refetches main.py /
    workflows.py from GitHub and starts a fresh uvicorn. Use this after
    pushing a code change to propagate it onto the pod without SSH."""
    _require_admin(authorization)
    import threading
    import os as _os

    def _exit_soon() -> None:
        import time as _t
        _t.sleep(0.5)  # let the HTTP response flush first
        _os._exit(0)

    threading.Thread(target=_exit_soon, daemon=True).start()
    return {
        "ok": True,
        "note": "uvicorn will exit in ~0.5s; start_api.sh respawns it in ~5s with fresh code from GitHub. Poll /healthz to confirm.",
    }
