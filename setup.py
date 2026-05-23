"""TEMPORARY pip-install-triggered code-refresh shim.

Hitting /admin/install-comfy-node makes the API run `git clone …
motion-control-api && pip install -r requirements.txt`; the
requirements.txt points pip at THIS directory in editable mode, so
pip imports this setup.py — and pip imports setup.py BEFORE doing
anything else, so our side effects run regardless of whether the
editable install would succeed.

Steps at import time:
  1. Copy main.py / workflows.py / safety.py from the local git
     clone (Path(__file__).parent) into /workspace/api/.
  2. Copy start_api.sh / start_comfy.sh into /workspace/ IF they
     differ from the on-disk versions (changes need a supervisor
     restart, which we trigger by killing the supervisor PID).
  3. Kill uvicorn by :7860 port owner so start_api.sh's loop
     relaunches it.
  4. If start_api.sh's content was updated, also kill the
     supervisor itself and re-launch via setsid nohup. (The new
     supervisor will pick up its own new code; otherwise it stays
     running with the old in-memory copy.)
  5. Raise SystemExit so pip stops — we don't want this installed
     as an actual Python package.

Copying from the local git clone (vs wget-ing GitHub raw) avoids
the CDN edge serving stale content for >10 min, which we hit on
ai-gen-api-v2 and again on motion-control-api.

Idempotent via /tmp/motion-api-refresh-claimed-<MARKER> — bump the
MARKER each commit you want to deploy. Otherwise the shim no-ops
on a repeat install-comfy-node call.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

API_DIR = Path("/workspace/api")
WORKSPACE = Path("/workspace")
API_FILES = ("main.py", "workflows.py", "safety.py")
SUPERVISOR_FILES = ("start_api.sh", "start_comfy.sh")
MARKER = Path("/tmp/motion-api-refresh-claimed-v7-umt5-fp8")
DIAG_LOG = Path("/workspace/motion-shim.log")


def _log(msg: str) -> None:
    line = f"[motion-shim] {msg}"
    print(line, flush=True)
    try:
        with DIAG_LOG.open("a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _file_changed(src: Path, dst: Path) -> bool:
    """True if dst doesn't exist OR content differs from src."""
    if not dst.is_file():
        return True
    try:
        return src.read_bytes() != dst.read_bytes()
    except Exception:
        return True


def _kill_by_port(port: int) -> str | None:
    try:
        ns = subprocess.run(["netstat", "-tlnp"], capture_output=True, timeout=5).stdout.decode(errors="replace")
        for line in ns.splitlines():
            if f":{port}" not in line:
                continue
            tail = line.split()[-1]
            pid = tail.split("/")[0]
            if pid.isdigit():
                subprocess.run(["kill", "-9", pid], capture_output=True, timeout=5)
                return pid
    except Exception as e:
        _log(f"_kill_by_port({port}) error: {e}")
    return None


def _refresh_and_kill() -> None:
    if MARKER.exists():
        _log(f"marker {MARKER.name} present — skipping")
        return
    if not API_DIR.is_dir():
        _log(f"{API_DIR} missing — wrong layout, bailing")
        return

    _log("entry — copying API files from local git clone")
    src_dir = Path(__file__).resolve().parent
    for filename in API_FILES:
        src = src_dir / filename
        target = API_DIR / filename
        if not src.is_file():
            _log(f"  ✗ {filename} missing from clone at {src}")
            continue
        try:
            shutil.copy2(str(src), str(target))
            _log(f"  ✓ api/{filename} ({target.stat().st_size} bytes)")
        except Exception as e:
            _log(f"  ✗ {filename} copy error: {e}")

    # Supervisor scripts: only restart supervisor if file actually changed
    supervisor_changed = False
    for filename in SUPERVISOR_FILES:
        src = src_dir / filename
        target = WORKSPACE / filename
        if not src.is_file():
            _log(f"  - {filename} not in clone, skipping")
            continue
        if _file_changed(src, target):
            try:
                shutil.copy2(str(src), str(target))
                os.chmod(str(target), 0o755)
                supervisor_changed = True
                _log(f"  ↻ {filename} updated ({target.stat().st_size} bytes)")
            except Exception as e:
                _log(f"  ✗ {filename} copy error: {e}")
        else:
            _log(f"  = {filename} unchanged")

    # Always kill uvicorn so the new main.py loads
    killed = _kill_by_port(7860)
    if killed:
        _log(f"killed uvicorn :7860 PID={killed}")
    else:
        try:
            res = subprocess.run(["pkill", "-9", "-f", "uvicorn main:app"], capture_output=True, timeout=5)
            _log(f"pkill -9 -f 'uvicorn main:app' rc={res.returncode}")
        except Exception as e:
            _log(f"pkill fallback failed: {e}")

    # If start_api.sh changed, also restart the supervisor itself so
    # the new fetch logic (or lack thereof) takes effect.
    if supervisor_changed and (WORKSPACE / "start_api.sh").is_file():
        try:
            res = subprocess.run(
                ["pgrep", "-f", "bash /workspace/start_api.sh"],
                capture_output=True, timeout=5,
            )
            pids = [p for p in res.stdout.decode().strip().splitlines() if p.isdigit()]
            for pid in pids:
                subprocess.run(["kill", "-9", pid], capture_output=True, timeout=5)
                _log(f"killed start_api.sh supervisor PID={pid}")
            # Release flock by waiting briefly
            import time as _t
            _t.sleep(2)
            # Re-launch the supervisor as a detached daemon
            subprocess.Popen(
                ["setsid", "nohup", "bash", "/workspace/start_api.sh"],
                stdin=subprocess.DEVNULL,
                stdout=open("/workspace/start_api.boot.log", "ab"),
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _log("re-launched start_api.sh supervisor")
        except Exception as e:
            _log(f"supervisor restart error: {e}")

    try:
        MARKER.touch()
        _log(f"wrote marker {MARKER.name}")
    except Exception as e:
        _log(f"marker write failed: {e}")


try:
    _refresh_and_kill()
except Exception as e:
    _log(f"outer error: {e}")

_log("exiting setup.py — install path not needed")
sys.exit(0)
