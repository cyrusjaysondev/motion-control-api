"""Wan 2.2 Animate workflow builder.

PLACEHOLDER for Phase 1 (repo scaffolding). The real workflow is
filled in during Phase 6 of the deploy plan, after the model files
are downloaded to the pod and we've confirmed which kijai
WanVideoWrapper node ids are stable.

Reference: kijai/ComfyUI-WanVideoWrapper example workflows in
  https://github.com/kijai/ComfyUI-WanVideoWrapper/tree/main/example_workflows

Target node graph (subject to refinement in Phase 6):

    CheckpointLoaderSimple / WanVideoModelLoader
         ↓
    WanVideoVAELoader, WanVideoTextEncoderLoader
         ↓                       ↓
    WanVideoTextEncode (positive, negative)
         ↓
    LoadImage (character) → resize → WanVideoEncode (identity latent)
    LoadVideo (driving)   → resize → WanVideoEncode (driving latent)
         ↓                              ↓
                  WanVideoSampler  (Wan 2.2 Animate 14B FP8)
                        ↓
                  WanVideoDecode → ColorMatch (drift mitigation)
                        ↓
                  VHS_VideoCombine → output mp4

The placeholder below raises NotImplementedError with a clear message
so /motion calls fail fast and visibly during scaffold-phase deploys.
"""


def build_wan_motion_workflow(
    reference_video_filename: str,
    character_image_filename: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    length: int,
    fps: int,
    seed: int,
) -> dict:
    """Build the Wan 2.2 Animate motion-control workflow.

    Phase 1 scaffold raises until Phase 6 fills in the real node graph.
    See module docstring for the target architecture.
    """
    raise NotImplementedError(
        "workflows.build_wan_motion_workflow is Phase 6 work. "
        "Repo scaffolding (Phase 1) is in place; the kijai/ComfyUI-WanVideoWrapper "
        "node graph + WanVideo model wiring need to be written after the pod "
        "is provisioned and the model weights are downloaded. See workflows.py "
        "module docstring for the target graph."
    )
