"""Content safety stub.

motion-control-api Phase 1 ships without content moderation. If we add
NSFW / face-block filtering later, the API surface should match
ai-gen-api-v2/safety.py (check_image, check_video) so callers can swap
backends without code changes.

This module exists so the setup.py shim's FILES_TO_REFRESH list always
has a target to wget — the shim treats missing files as errors.
"""


def check_image(*args, **kwargs):
    return {"ok": True, "reason": "safety stub — motion-control-api Phase 1"}


def check_video(*args, **kwargs):
    return {"ok": True, "reason": "safety stub — motion-control-api Phase 1"}
