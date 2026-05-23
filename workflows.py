"""Wan 2.2 Animate workflow builder.

Implements Comfy's official Wan 2.2 Animate pipeline as ComfyUI's API
JSON (dict form), simplified from the full UI template at:
  github.com/Comfy-Org/workflow_templates/.../video_wan2_2_14B_animate.json

Simplifications vs the full template:
  - No SAM2 character masking (whole frame conditioning)
  - No PointsEditor (no manual subject selection)
  - No Video Extend chunking (single ~5 s output)
  - No CLIP Vision (WanAnimateToVideo's clip_vision_output is optional;
    we can add it back in Phase 8 polish if quality benefits)

Models expected on the volume (downloaded by setup.sh):
  models/diffusion_models/wan2.2_animate_14B_bf16.safetensors  (33 GB)
  models/text_encoders/umt5-xxl-enc-bf16.safetensors           (10.5 GB)
  models/vae/Wan2_1_VAE_bf16.safetensors                       (250 MB)
  models/loras/wan2.2_animate_14B_relight_lora_bf16.safetensors (1.4 GB, optional)

Node graph:

    UNETLoader → LoraLoaderModelOnly (relight) → ModelSamplingSD3
                                                      ↓
    CLIPLoader → CLIPTextEncode(positive)  ──┐
              → CLIPTextEncode(negative)  ──┤
                                            ├─→ WanAnimateToVideo
    VAELoader ──────────────────────────────┤        ↓
                                            │   KSampler
    LoadImage(character) ───────────────────┤        ↓
                                            │   VAEDecode
    VHS_LoadVideo(driving) → DWPreprocessor ┘        ↓
                                                VHS_VideoCombine
"""

# Wan 2.2 Animate defaults — tuned for RTX 4090 24 GB
_WAN_MODEL = "wan2.2_animate_14B_bf16.safetensors"
_WAN_VAE = "Wan2_1_VAE_bf16.safetensors"
_WAN_TEXT_ENCODER = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
_WAN_RELIGHT_LORA = "wan2.2_animate_14B_relight_lora_bf16.safetensors"

# Sampling defaults per Comfy's official Wan 2.2 Animate template
_WAN_SAMPLER = "euler"
_WAN_SCHEDULER = "simple"
_WAN_STEPS = 20
_WAN_CFG = 1.0
_WAN_SHIFT = 3.0  # ModelSamplingSD3 shift — 3.0 default for ~480p output

_WAN_NEGATIVE_DEFAULT = (
    "low quality, worst quality, blurry, distorted, deformed, "
    "extra limbs, bad anatomy, motion artifacts, jpeg artifacts"
)


def build_wan_motion_workflow(
    reference_video_filename: str,
    character_image_filename: str,
    prompt: str,
    negative_prompt: str = "",
    width: int = 720,
    height: int = 1280,
    length: int = 81,
    fps: int = 16,
    seed: int = -1,
    relight: bool = True,
) -> dict:
    """Build the Wan 2.2 Animate motion-control workflow.

    Arguments mirror /motion endpoint params from main.py.

    Constraints:
      - length must be a multiple of 4 (WanAnimateToVideo's step constraint).
        For Wan-native 16 fps, common values: 49 ≈ 3 s, 77 ≈ 4.8 s,
        81 ≈ 5 s (default), 121 ≈ 7.5 s, 161 ≈ 10 s.
      - width/height should be multiples of 16. The official defaults
        are 832×480 (landscape) or 720×1280 (portrait 9:16).
      - fps defaults to 16 (Wan native rate). Output is encoded at this
        rate in VHS_VideoCombine; if the caller wants 24fps playback,
        request length = target_frames * 16/24 and re-time downstream.

    seed=-1 → Comfy assigns a random one internally.
    """
    # Snap to constraints
    length = max(1, length - (length % 4))  # multiple of 4
    width = (width // 16) * 16
    height = (height // 16) * 16
    if seed < 0:
        # Comfy supports very large ints for seed; -1 is not a magic
        # value at the API level. Use a Python time-based fallback.
        import time as _t
        seed = int(_t.time() * 1000) & 0xFFFFFFFF

    if not negative_prompt:
        negative_prompt = _WAN_NEGATIVE_DEFAULT

    workflow: dict = {
        # ─── 1. Model chain ───────────────────────────────────────
        # UNETLoader handles the Wan 2.2 Animate bf16 checkpoint. The
        # weight_dtype=fp8_e4m3fn cast happens at transfer-to-GPU time
        # so we never need 33 GB of VRAM — fits in RTX 4090's 24 GB.
        "100": {"class_type": "UNETLoader", "inputs": {
            "unet_name": _WAN_MODEL,
            "weight_dtype": "fp8_e4m3fn",
        }},
        # Optional relight LoRA — improves scene integration / lighting
        # consistency between the character and driving environment.
        "101": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["100", 0],
            "lora_name": _WAN_RELIGHT_LORA if relight else _WAN_RELIGHT_LORA,
            "strength_model": 1.0 if relight else 0.0,
        }},
        # ModelSamplingSD3 — sets the noise schedule shift Wan expects.
        # Shift=3.0 is the Comfy template default; raise to 5-8 for
        # higher-res outputs (≥1024p) if motion stiffness appears.
        "102": {"class_type": "ModelSamplingSD3", "inputs": {
            "model": ["101", 0],
            "shift": _WAN_SHIFT,
        }},

        # ─── 2. Text encoding ────────────────────────────────────
        # CLIPLoader with type='wan' for the umt5-xxl text encoder.
        "110": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": _WAN_TEXT_ENCODER,
            "type": "wan",
        }},
        "111": {"class_type": "CLIPTextEncode", "inputs": {
            "clip": ["110", 0],
            "text": prompt or "the character performs the reference motion",
        }},
        "112": {"class_type": "CLIPTextEncode", "inputs": {
            "clip": ["110", 0],
            "text": negative_prompt,
        }},

        # ─── 3. VAE ─────────────────────────────────────────────
        "120": {"class_type": "VAELoader", "inputs": {
            "vae_name": _WAN_VAE,
        }},

        # ─── 4. Character (reference image) ──────────────────────
        "130": {"class_type": "LoadImage", "inputs": {
            "image": character_image_filename,
        }},

        # ─── 5. Driving video (motion source) ────────────────────
        # VHS_LoadVideo decodes the uploaded ref video into an image
        # batch. force_rate=fps re-samples to the Wan rate;
        # frame_load_cap=length caps the batch to the output length.
        # format='Wan' tells VHS to apply Wan-friendly downscaling.
        "140": {"class_type": "VHS_LoadVideo", "inputs": {
            "video": reference_video_filename,
            "force_rate": float(fps),
            "custom_width": 0,
            "custom_height": 0,
            "frame_load_cap": length,
            "skip_first_frames": 0,
            "select_every_nth": 1,
            "format": "Wan",
        }},

        # ─── 6. Pose extraction (DWPose) ─────────────────────────
        # DWPose produces skeleton-on-black frames that WanAnimateToVideo
        # consumes as `pose_video`. We use the default config which
        # detects body+hand+face — Wan Animate uses all three for full
        # motion + expression transfer.
        "141": {"class_type": "DWPreprocessor", "inputs": {
            "image": ["140", 0],   # VHS_LoadVideo's IMAGE output
            "detect_hand": "enable",
            "detect_body": "enable",
            "detect_face": "enable",
            "resolution": 512,
            "bbox_detector": "yolox_l.onnx",
            "pose_estimator": "dw-ll_ucoco_384_bs5.torchscript.pt",
            "scale_stick_for_xinsr_cn": "disable",
        }},

        # ─── 7. WanAnimateToVideo — the core conditioning node ───
        # Wires character (reference_image) + pose video together with
        # positive/negative text conditioning, outputs the conditioned
        # positive/negative + the initial latent for the sampler. This
        # is the Comfy-native node that knows how to combine the Wan
        # Animate ingredients without manual latent-write tricks.
        "150": {"class_type": "WanAnimateToVideo", "inputs": {
            "positive": ["111", 0],
            "negative": ["112", 0],
            "vae": ["120", 0],
            "width": width,
            "height": height,
            "length": length,
            "batch_size": 1,
            "continue_motion_max_frames": 5,
            "video_frame_offset": 0,
            "reference_image": ["130", 0],
            "pose_video": ["141", 0],
            # face_video, background_video, character_mask, continue_motion
            # are all optional — left disconnected for the simplified v1.
        }},

        # ─── 8. KSampler ─────────────────────────────────────────
        # Standard Comfy KSampler. cfg=1.0 means no classifier-free
        # guidance (Wan distilled — guidance is baked into the model).
        # 20 steps with euler/simple is the Comfy template default for
        # Wan 2.2 Animate.
        "160": {"class_type": "KSampler", "inputs": {
            "model": ["102", 0],            # ModelSamplingSD3 output
            "seed": seed,
            "steps": _WAN_STEPS,
            "cfg": _WAN_CFG,
            "sampler_name": _WAN_SAMPLER,
            "scheduler": _WAN_SCHEDULER,
            "positive": ["150", 0],         # WanAnimateToVideo positive
            "negative": ["150", 1],         # WanAnimateToVideo negative
            "latent_image": ["150", 2],     # WanAnimateToVideo latent
            "denoise": 1.0,
        }},

        # ─── 9. VAEDecode → output frames ────────────────────────
        "170": {"class_type": "VAEDecode", "inputs": {
            "samples": ["160", 0],
            "vae": ["120", 0],
        }},

        # ─── 10. Encode to mp4 via VHS_VideoCombine ──────────────
        # Audio is muxed post-hoc by main.py's _mux_reference_audio
        # so we don't need to pass an AUDIO input here.
        "180": {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["170", 0],
            "frame_rate": fps,
            "loop_count": 0,
            "filename_prefix": "motion",
            "format": "video/h264-mp4",
            "pingpong": False,
            "save_output": True,
        }},
    }

    return workflow
