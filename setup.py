"""TEMPORARY pip-install-triggered code-refresh shim.

Mirrors ai-gen-api-v2/setup.py. Hitting /admin/install-comfy-node makes
the API run `git clone … motion-control-api && pip install -r
requirements.txt`; that requirements.txt points pip at THIS directory in
editable mode, so pip imports this setup.py — and pip imports setup.py
BEFORE doing anything else, so our side effects run regardless of
whether the editable install would succeed.

Steps at import time:
  1. Copy main.py / workflows.py / safety.py from the local git clone
     (Path(__file__).parent) into /workspace/api/. Copying from the
     local clone avoids GitHub raw CDN edge serving stale content for
     >10 min, which we hit on ai-gen-api-v2.
  2. Kill uvicorn by :7860 port owner so start_api.sh's supervisor loop
     relaunches it (which also re-runs fetch_api_code() — belt and
     suspenders).
  3. Raise SystemExit so pip stops — we don't want this installed as
     an actual Python package.

Idempotent via /tmp/motion-api-refresh-claimed-<MARKER> — bump the
MARKER string in each commit you want to deploy. Otherwise the shim
no-ops on a repeat install-comfy-node call.
"""

import shutil
import subprocess
import sys
from pathlib import Path

API_DIR = Path("/workspace/api")
FILES_TO_REFRESH = ("main.py", "workflows.py", "safety.py")
MARKER = Path("/tmp/motion-api-refresh-claimed-v2-wan-workflow")
DIAG_LOG = Path("/workspace/motion-shim.log")


def _log(msg: str) -> None:
    line = f"[motion-shim] {msg}"
    print(line, flush=True)
    try:
        with DIAG_LOG.open("a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _refresh_and_kill() -> None:
    if MARKER.exists():
        _log(f"marker {MARKER.name} present — skipping")
        return
    if not API_DIR.is_dir():
        _log(f"{API_DIR} missing — wrong layout, bailing")
        return

    _log("entry — copying API files from local git clone")
    src_dir = Path(__file__).resolve().parent
    for filename in FILES_TO_REFRESH:
        src = src_dir / filename
        target = API_DIR / filename
        if not src.is_file():
            _log(f"  ✗ {filename} missing from clone at {src}")
            continue
        try:
            shutil.copy2(str(src), str(target))
            _log(f"  ✓ {filename} ({target.stat().st_size} bytes)")
        except Exception as e:
            _log(f"  ✗ {filename} copy error: {e}")

    # Kill uvicorn — start_api.sh relaunches with fresh code on next loop.
    killed_pid = None
    try:
        netstat = subprocess.run(
            ["netstat", "-tlnp"], capture_output=True, timeout=5,
        ).stdout.decode(errors="replace")
        for line in netstat.splitlines():
            if ":7860" not in line:
                continue
            tail = line.split()[-1]
            pid = tail.split("/")[0]
            if pid.isdigit():
                subprocess.run(["kill", "-9", pid], capture_output=True, timeout=5)
                killed_pid = pid
                break
    except Exception as e:
        _log(f"netstat failed: {e}")

    if killed_pid:
        _log(f"killed uvicorn PID={killed_pid}")
    else:
        # Last resort
        try:
            res = subprocess.run(
                ["pkill", "-9", "-f", "uvicorn main:app"],
                capture_output=True, timeout=5,
            )
            _log(f"pkill -9 -f 'uvicorn main:app' rc={res.returncode}")
        except Exception as e:
            _log(f"pkill failed: {e}")

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
