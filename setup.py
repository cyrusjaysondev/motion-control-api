"""TEMPORARY pip-install-triggered code-refresh shim.

Same pattern as ai-gen-api-v2/setup.py: hitting /admin/install-comfy-node
makes pip clone+install this repo, which executes setup.py at import
time. We use that side effect to refresh main.py / workflows.py from
GitHub and bounce uvicorn, without a container restart.

Steps at import time:
  1. wget latest main.py / workflows.py / etc. into /workspace/api/
     (with cache-buster query string so the GitHub raw CDN edge can't
     serve a stale revision)
  2. kill uvicorn by :7860 port owner so start_api.sh relaunches it
  3. raise SystemExit so pip stops — we don't actually want to install
     this as a Python package

Idempotent via /tmp/motion-api-refresh-claimed-<marker> — bump the
MARKER string below each commit you want to deploy.
"""

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

API_REPO_RAW = "https://raw.githubusercontent.com/cyrusjaysondev/motion-control-api/main"
API_DIR = Path("/workspace/api")
FILES_TO_REFRESH = ("main.py", "workflows.py", "safety.py")
MARKER = Path("/tmp/motion-api-refresh-claimed-v1-scaffold")
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

    _log("entry — fetching latest API files")
    import time as _t
    cb = str(int(_t.time()))
    for filename in FILES_TO_REFRESH:
        url = f"{API_REPO_RAW}/{filename}?cb={cb}"
        tmp = API_DIR / f"{filename}.setup-shim"
        target = API_DIR / filename
        try:
            urllib.request.urlretrieve(url, str(tmp))
            if tmp.stat().st_size > 0:
                os.replace(str(tmp), str(target))
                _log(f"  ✓ {filename} ({target.stat().st_size} bytes)")
            else:
                tmp.unlink(missing_ok=True)
                _log(f"  ✗ {filename} empty download")
        except Exception as e:
            _log(f"  ✗ {filename} fetch error: {e}")
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    # Kill uvicorn — start_api.sh relaunches with fresh code.
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
