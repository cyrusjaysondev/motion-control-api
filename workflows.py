"""Wan 2.2 Animate motion-transfer workflow — AIGCTV/kijai port.

Replaces the Comfy-native WanAnimateToVideo graph with kijai's
ComfyUI-WanVideoWrapper graph + the Lightning 4-step LoRA, lifting
the upstream design from the AIGCTV walkthrough
(https://www.youtube.com/watch?v=ebgoSXDFcZw).

Why the swap:
  - 4 steps instead of 20 (Lightning LoRA) → ~5× faster sampling
  - CLIP Vision encoding of the character image → better identity hold
  - VitPose-L wholebody pose extraction → finer hands/face motion
  - First-frame-prepend warm-up → less identity drift in the first frames
  - Context windows → handles 5–15 s clips without OOM
  - Block-swap (offload img/txt embeds) → 14 B model fits in 24 GB

Node graph (Wan-wrapper variants in CAPS):

  WANVIDEOMODELLOADER ─┐
                       ├─ SetLoRAs (relight + Lightning) ─ SetBlockSwap ─┐
                       │                                                │
  WANVIDEOTEXTENCODE   ────────────────────────────────────────────────┐│
                                                                       ││
  CLIPVISIONLOADER ─ WanVideoClipVisionEncode(character) ──────┐       ││
                                                               │       ││
  LoadImage(character) ─ ImageResizeKJv2 ──────────────┐       │       ││
                                                       │       │       ││
  VHS_LoadVideo(ref) ─ ImageFromBatch(first) ─┐        │       │       ││
                                              ├─ ImageBatch ─ Pose ─┐  ││
                                              │            (Onnx)   │  ││
                                              └────────────────────│┐  ││
                                                                   ││  ││
                       WanVideoAnimateEmbeds (ref + pose + face + clip) ─┐
                                                                         │
                                            WANVIDEOSAMPLER ─ Decode ─ ImageFromBatch(skip warm) ─ VHS_VideoCombine

Models on the volume:
  models/diffusion_models/Wan2_2-Animate-14B_fp8_scaled_e4m3fn_KJ_v2.safetensors
  models/vae/Wan2_1_VAE_fp32.safetensors
  models/text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors
  models/clip_vision/clip_vision_h.safetensors
  models/loras/Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors
  models/loras/WanAnimate_relight_lora_fp16.safetensors
  models/detection/vitpose-l-wholebody.onnx
  models/detection/yolov10m.onnx
"""

# ─── Model filenames on the network volume ───────────────────────
_WAN_MODEL = "Wan2_2-Animate-14B_fp8_scaled_e4m3fn_KJ_v2.safetensors"
_WAN_VAE = "Wan2_1_VAE_fp32.safetensors"
_WAN_TEXT_ENCODER = "umt5-xxl-enc-fp8_e4m3fn.safetensors"
_WAN_CLIP_VISION = "clip_vision_h.safetensors"
_WAN_RELIGHT_LORA = "WanAnimate_relight_lora_fp16.safetensors"
_WAN_LIGHTNING_LORA = "Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors"
_VITPOSE_MODEL = "vitpose-l-wholebody.onnx"
_YOLO_MODEL = "yolov10m.onnx"

# ─── Sampler defaults (AIGCTV walkthrough) ────────────────────────
_WAN_STEPS = 4            # Lightning LoRA lets us run in 4 steps
_WAN_CFG = 1.0            # distilled model — guidance is baked in
_WAN_SHIFT = 5.0          # noise schedule shift; 5.0 fits Lightning
_WAN_SCHEDULER = "dpm++_sde"
_WAN_BLOCKS_TO_SWAP = 2   # offloaded transformer blocks for VRAM

# Context windowing — lets the sampler handle clips longer than the
# single chunk size by sliding a window through the latent. With
# context_frames=81 and overlap=16, a 161-frame clip gets sampled in
# two overlapping chunks. Bump context_frames to 161 for longer clips
# on cards with more VRAM.
_WAN_CONTEXT_FRAMES = 81
_WAN_CONTEXT_STRIDE = 4
_WAN_CONTEXT_OVERLAP = 16

_WAN_NEGATIVE_DEFAULT = (
    "low quality, worst quality, blurry, distorted, deformed, "
    "extra limbs, bad anatomy, motion artifacts, jpeg artifacts, "
    "static, frozen frames"
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
    lightning_steps: int = _WAN_STEPS,
    sampler_shift: float = _WAN_SHIFT,
) -> dict:
    """Build the AIGCTV-style Wan 2.2 Animate API workflow.

    Args mirror /motion's form fields:

      reference_video_filename  driving motion video (any common format)
      character_image_filename  the identity source (PNG/JPG)
      prompt                    scene description, optional
      negative_prompt           defaults to a generic degradation list
      width / height            output canvas in pixels (mult. of 16)
      length                    output frame count (multiple of 4)
      fps                       output frame rate
      seed                      -1 for random
      relight                   apply the relight LoRA
      lightning_steps           sampler steps (4 = Lightning default)
      sampler_shift             noise-schedule shift (5 = Lightning default)

    Returns a Comfy API-format dict ready for POST /prompt.

    Length math: the workflow PREPENDS `length` warm-up frames (copies of
    the ref video's first frame) before the actual motion, samples 2×
    length frames, then keeps only the second half. So the user-visible
    output is `length` frames.
    """
    # Snap inputs to model constraints
    length = max(4, length - (length % 4))
    width = (width // 16) * 16
    height = (height // 16) * 16
    if seed < 0:
        import time as _t
        seed = int(_t.time() * 1000) & 0xFFFFFFFF

    if not negative_prompt:
        negative_prompt = _WAN_NEGATIVE_DEFAULT

    # All node IDs are arbitrary strings — kept stable for readability.
    workflow: dict = {
        # ─── 1. Model + LoRA + block-swap chain ───────────────────
        # Loads the fp8_scaled KJ_v2 Wan checkpoint with sageattn for
        # speed. base_precision=fp16_fast keeps math compatible with
        # the Lightning LoRA weights, which were distilled at fp16.
        "100": {"class_type": "WanVideoModelLoader", "inputs": {
            "model": _WAN_MODEL,
            "base_precision": "fp16_fast",
            "quantization": "fp8_e4m3fn_scaled",
            "load_device": "offload_device",
            "attention_mode": "sageattn",
            "rms_norm_function": "default",
        }},

        # LoRA stack: relight LoRA (optional, for lighting consistency)
        # + Lightning LoRA (cuts sampling to 4 steps). Empty slots are
        # required by WanVideoLoraSelectMulti's 5-slot schema.
        "101": {"class_type": "WanVideoLoraSelectMulti", "inputs": {
            "lora_0": _WAN_RELIGHT_LORA if relight else "none",
            "strength_0": 1.0 if relight else 0.0,
            "lora_1": _WAN_LIGHTNING_LORA,
            "strength_1": 1.0,
            "lora_2": "none", "strength_2": 1.0,
            "lora_3": "none", "strength_3": 1.0,
            "lora_4": "none", "strength_4": 1.0,
            "low_mem_load": False,
            "merge_loras": False,
        }},
        "102": {"class_type": "WanVideoSetLoRAs", "inputs": {
            "model": ["100", 0],
            "lora": ["101", 0],
        }},

        # Block swap config — offloads `blocks_to_swap` transformer
        # blocks + the image/text embedders to CPU between steps. This
        # is what lets the 14 B fp8 model fit in 24 GB VRAM with room
        # for VAE + CLIP Vision.
        "103": {"class_type": "WanVideoBlockSwap", "inputs": {
            "blocks_to_swap": _WAN_BLOCKS_TO_SWAP,
            "offload_img_emb": True,
            "offload_txt_emb": True,
            "use_non_blocking": False,
            "vace_blocks_to_swap": 0,
            "prefetch_blocks": 1,
            "block_swap_debug": False,
        }},
        "104": {"class_type": "WanVideoSetBlockSwap", "inputs": {
            "model": ["102", 0],
            "block_swap_args": ["103", 0],
        }},

        # ─── 2. VAE ───────────────────────────────────────────────
        "110": {"class_type": "WanVideoVAELoader", "inputs": {
            "model_name": _WAN_VAE,
            "precision": "bf16",
            "use_cpu_cache": False,
            "verbose": False,
        }},

        # ─── 3. Text encoding (cached on first call) ───────────────
        # WanVideoTextEncodeCached encodes both positive and negative
        # in one node and caches the umt5-xxl encoder result so
        # subsequent jobs with the same prompts skip the encode.
        "120": {"class_type": "WanVideoTextEncodeCached", "inputs": {
            "model_name": _WAN_TEXT_ENCODER,
            "precision": "bf16",
            "positive_prompt": prompt or "the character performs the reference motion",
            "negative_prompt": negative_prompt,
            "quantization": "fp8_e4m3fn",
            "use_disk_cache": False,
            "device": "gpu",
        }},

        # ─── 4. CLIP Vision (identity encoding) ────────────────────
        "130": {"class_type": "CLIPVisionLoader", "inputs": {
            "clip_name": _WAN_CLIP_VISION,
        }},

        # ─── 5. Character image (identity reference) ───────────────
        "140": {"class_type": "LoadImage", "inputs": {
            "image": character_image_filename,
        }},
        # Resize+crop the character to the target canvas. crop+top keeps
        # the face when shrinking portraits; divisible_by=8 satisfies
        # Wan's patch-embedding stride.
        "141": {"class_type": "ImageResizeKJv2", "inputs": {
            "width": width,
            "height": height,
            "upscale_method": "lanczos",
            "keep_proportion": "crop",
            "pad_color": "0, 0, 0",
            "crop_position": "top",
            "divisible_by": 8,
            "device": "cpu",
            "image": ["140", 0],
        }},
        # CLIP Vision encoding of the resized character. combine_embeds
        # = average is robust when there's only one input image.
        "142": {"class_type": "WanVideoClipVisionEncode", "inputs": {
            "strength_1": 1.0,
            "strength_2": 1.0,
            "crop": "disabled",
            "combine_embeds": "average",
            "force_offload": True,
            "tiles": 0,
            "ratio": 0.5,
            "clip_vision": ["130", 0],
            "image_1": ["141", 0],
        }},

        # ─── 6. Driving video → warm-up batch ──────────────────────
        # Load the ffmpeg-normalized ref video. force_rate=fps re-samples
        # to the Wan rate; frame_load_cap caps at `length` so we never
        # sample more motion than we asked for.
        "150": {"class_type": "VHS_LoadVideo", "inputs": {
            "video": reference_video_filename,
            "force_rate": float(fps),
            "force_size": "Disabled",
            "custom_width": 0,
            "custom_height": 0,
            "frame_load_cap": length,
            "skip_first_frames": 0,
            "select_every_nth": 1,
            "format": "AnimateDiff",
        }},
        # First-frame-prepend warm-up: take the first frame, repeat it
        # `length` times, then concat with the actual video. The model
        # uses these warm-up frames to lock onto the character identity
        # before applying motion; we discard them post-decode. Total
        # batch size = 2 * length.
        "151": {"class_type": "ImageFromBatch", "inputs": {
            "batch_index": 0,
            "length": 1,
            "image": ["150", 0],
        }},
        "152": {"class_type": "RepeatImageBatch", "inputs": {
            "amount": length,
            "image": ["151", 0],
        }},
        "153": {"class_type": "ImageBatch", "inputs": {
            "image1": ["152", 0],
            "image2": ["150", 0],
        }},

        # ─── 7. Pose + face extraction (VitPose-L wholebody) ───────
        # OnnxDetectionModelLoader loads YOLO (bbox detect) + ViTPose
        # (keypoint regression) onto the CUDA execution provider so
        # detection happens on GPU. PoseAndFaceDetection runs the full
        # pipeline over the 2×length frame batch, emitting:
        #   slot 0: pose-skeleton frames (what Wan conditions on)
        #   slot 1: face crops (face_images conditioning)
        "160": {"class_type": "OnnxDetectionModelLoader", "inputs": {
            "vitpose_model": _VITPOSE_MODEL,
            "yolo_model": _YOLO_MODEL,
            "onnx_device": "CUDAExecutionProvider",
        }},
        "161": {"class_type": "PoseAndFaceDetection", "inputs": {
            "width": width,
            "height": height,
            "model": ["160", 0],
            "images": ["153", 0],
        }},
        # PoseAndFaceDetection emits POSEDATA at slot 0, not IMAGE. The
        # WanVideoAnimateEmbeds.pose_images input wants drawn skeleton
        # frames, so route the POSEDATA through DrawViTPose first. The
        # stick widths at -1 mean "auto-scale" based on canvas size,
        # which is what the AIGCTV reference graph uses.
        "162": {"class_type": "DrawViTPose", "inputs": {
            "width": width,
            "height": height,
            "retarget_padding": 16,
            "body_stick_width": -1,
            "hand_stick_width": -1,
            "draw_head": True,
            "pose_data": ["161", 0],
        }},

        # ─── 8. WanVideoAnimateEmbeds (combine all conditionings) ──
        # The Wan-wrapper analog of Comfy's WanAnimateToVideo. Takes
        # the VAE, CLIP-Vision embeds, character ref, pose video, and
        # face crops, returns an image-conditioning bundle that feeds
        # the sampler. num_frames + frame_window_size both = 2 * length
        # because we're sampling the prepended warm-up too.
        "170": {"class_type": "WanVideoAnimateEmbeds", "inputs": {
            "width": width,
            "height": height,
            "num_frames": 2 * length,
            "force_offload": True,
            "frame_window_size": 2 * length,
            "colormatch": "disabled",
            "pose_strength": 1.0,
            "face_strength": 1.0,
            "tiled_vae": False,
            "vae": ["110", 0],
            "clip_embeds": ["142", 0],
            "ref_images": ["141", 0],
            "pose_images": ["162", 0],   # DrawViTPose IMAGE output
            "face_images": ["161", 1],   # PoseAndFaceDetection face crops
        }},

        # ─── 9. Context options (sliding-window sampling) ──────────
        # Enables clip lengths beyond a single chunk by sampling
        # overlapping windows then blending. freenoise=True keeps the
        # noise pattern consistent across windows (less seam-flicker).
        "180": {"class_type": "WanVideoContextOptions", "inputs": {
            "context_schedule": "uniform_standard",
            "context_frames": _WAN_CONTEXT_FRAMES,
            "context_stride": _WAN_CONTEXT_STRIDE,
            "context_overlap": _WAN_CONTEXT_OVERLAP,
            "freenoise": True,
            "verbose": False,
            "fuse_method": "linear",
        }},

        # ─── 10. WanVideoSampler ───────────────────────────────────
        # Lightning LoRA + dpm++_sde + shift=5 = AIGCTV's recommended
        # combo. cfg=1 because distilled (CFG baked into the weights).
        # denoise_strength=1 means full denoise from pure noise.
        "190": {"class_type": "WanVideoSampler", "inputs": {
            "steps": lightning_steps,
            "cfg": _WAN_CFG,
            "shift": sampler_shift,
            "seed": seed,
            "force_offload": True,
            "scheduler": _WAN_SCHEDULER,
            "riflex_freq_index": 0,
            "denoise_strength": 1.0,
            "batched_cfg": False,
            "rope_function": "comfy",
            "start_step": 0,
            "end_step": -1,
            "add_noise_to_samples": False,
            "model": ["104", 0],
            "image_embeds": ["170", 0],
            "text_embeds": ["120", 0],
            "context_options": ["180", 0],
        }},

        # ─── 11. Decode latent → frames ────────────────────────────
        "200": {"class_type": "WanVideoDecode", "inputs": {
            "enable_vae_tiling": False,
            "tile_x": 272,
            "tile_y": 272,
            "tile_stride_x": 144,
            "tile_stride_y": 128,
            "normalization": "default",
            "vae": ["110", 0],
            "samples": ["190", 0],
        }},

        # ─── 12. Drop the warm-up half, keep the real output ───────
        # We sampled 2*length frames; the first `length` were warm-up
        # frames driven by a static pose. Discard them so the user-
        # visible output is exactly `length` motion frames.
        "210": {"class_type": "ImageFromBatch", "inputs": {
            "batch_index": length,
            "length": length,
            "image": ["200", 0],
        }},

        # ─── 13. Encode to mp4 ─────────────────────────────────────
        # Audio is muxed post-hoc by main.py's _mux_reference_audio,
        # so no audio input here.
        "220": {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["210", 0],
            "frame_rate": fps,
            "loop_count": 0,
            "filename_prefix": "motion",
            "format": "video/h264-mp4",
            "pix_fmt": "yuv420p",
            "crf": 19,
            "save_metadata": False,
            "trim_to_audio": False,
            "pingpong": False,
            "save_output": True,
            "no_preview": False,
        }},
    }

    return workflow
