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

        ws_url = f"ws://127.0.0.1:8188/ws?clientId={client_id}"
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
                        break

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
        jobs[job_id] = {
            **jobs[job_id],
            "status": "failed",
            "error": str(e),
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
    )

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


@app.get("/admin/disk-status")
async def admin_disk_status(authorization: str = Header(default=None)):
    """Run df + a couple of du checks so we can see what the kernel
    thinks the filesystem looks like — used when /motion fails with
    'Disk quota exceeded (os error 122)' and we can't SSH in."""
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
    return results


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
