"""
LTX-2 Video Generation UI (lt1.py)

Gradio-based web interface for LTX-2 video generation with:
- Two-stage pipeline (low-res + hi-res refinement)
- Joint audio-video generation
- Image conditioning (I2V)
- LoRA support
- Memory optimization (offload, FP8, block swap)
- Prompt enhancement with Gemma
"""

import os
import sys
import signal

# Try to use local patched gradio from modules/ if available (allows jobs to continue after browser disconnect)
_modules_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules")
if os.path.exists(_modules_path):
    sys.path.insert(0, _modules_path)

import gradio as gr
from gradio import themes
from gradio.themes.utils import colors
import time
import random
import subprocess
import re
from typing import Generator, List, Tuple, Optional, Dict
import threading
import json
from PIL import Image
from pathlib import Path
import tempfile
import shutil

# Global state
stop_event = threading.Event()
current_process = None
current_output_filename = None

# Defaults configuration
UI_CONFIGS_DIR = "ui_configs"
LT1_DEFAULTS_FILE = os.path.join(UI_CONFIGS_DIR, "lt1_defaults.json")


# =============================================================================
# Progress Parsing
# =============================================================================

def parse_ltx_progress_line(line: str) -> Optional[str]:
    """Parse LTX-2 generation progress output."""
    line = line.strip()

    # LTX-specific progress messages
    if ">>> Loading text encoder" in line:
        return "Loading text encoder..."
    if ">>> Enhancing prompt" in line:
        return "Enhancing prompt with Gemma..."
    if ">>> Encoding prompts" in line:
        return "Encoding text prompts..."
    # One-stage pipeline
    if ">>> One-stage: Loading" in line:
        return "One-stage: Loading models..."
    if ">>> One-stage: Generating" in line:
        return "One-stage: Generating at full resolution..."
    if ">>> One-stage completed" in line:
        match = re.search(r'(\d+\.?\d*)s', line)
        if match:
            return f"One-stage completed ({match.group(1)}s)"
    # Refine-only pipeline
    if ">>> Refine-only mode" in line:
        return "Refine-only: Loading input video..."
    if ">>> Input video encoded" in line:
        match = re.search(r'(\d+\.?\d*)s', line)
        if match:
            return f"Video encoded ({match.group(1)}s)"
    # Two-stage pipeline
    if ">>> Stage 1: Loading" in line:
        return "Stage 1: Loading models..."
    if ">>> Stage 1: Generating" in line:
        return "Stage 1: Low-res generation..."
    if ">>> Stage 1 completed" in line:
        match = re.search(r'(\d+\.?\d*)s', line)
        if match:
            return f"Stage 1 completed ({match.group(1)}s)"
    if ">>> Upsampling" in line:
        return "Upsampling latents (2x)..."
    if ">>> Stage 2: Loading" in line:
        return "Stage 2: Loading refined model..."
    if ">>> Stage 2: Refining" in line:
        return "Stage 2: Hi-res refinement..."
    if ">>> Stage 2 completed" in line:
        match = re.search(r'(\d+\.?\d*)s', line)
        if match:
            return f"Stage 2 completed ({match.group(1)}s)"
    # Stage 3 progress
    if ">>> Stage 3: Loading" in line:
        return "Stage 3: Loading transformer..."
    if ">>> Stage 3: Refining" in line:
        return "Stage 3: Final refinement..."
    if ">>> Stage 3 completed" in line:
        match = re.search(r'(\d+\.?\d*)s', line)
        if match:
            return f"Stage 3 completed ({match.group(1)}s)"
    if ">>> Linear mode: skipping upsampling" in line:
        return "Linear mode: same resolution..."
    if ">>> Decoding video" in line:
        return "Decoding video from latents..."
    if ">>> Decoding audio" in line:
        return "Decoding audio..."
    if ">>> Encoding video" in line:
        return "Encoding video to MP4..."
    # AV Extension mode
    if "AV Extension Mode" in line:
        return "AV Extension: Starting..."
    if "[AV Extension] Video mask" in line:
        return "AV Extension: Creating video mask..."
    if "[AV Extension] Audio mask" in line:
        return "AV Extension: Creating audio mask..."
    if ">>> Running masked denoising" in line:
        return "AV Extension: Masked denoising..."
    if ">>> Creating extended latent" in line:
        return "AV Extension: Creating extended latent space..."
    if ">>> Extracting audio from input" in line:
        return "AV Extension: Extracting audio..."
    if ">>> Encoding video to latent" in line:
        return "AV Extension: Encoding video to latent..."
    if ">>> Done!" in line:
        return "Generation complete!"
    if ">>> Total generation time" in line:
        match = re.search(r'(\d+\.?\d*)s', line)
        if match:
            return f"Total time: {match.group(1)}s"

    # TQDM progress bar parsing
    match = re.search(r'(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[.*?<([\d:]+)', line)
    if match:
        percent = match.group(1)
        current = match.group(2)
        total = match.group(3)
        eta = match.group(4)
        return f"Denoising: {percent}% ({current}/{total}) ETA: {eta}"

    return None


# =============================================================================
# LoRA Discovery
# =============================================================================

def get_ltx_lora_options(lora_folder: str = "lora") -> List[str]:
    """
    Discover LTX LoRAs in the specified folder.
    Returns ['None'] + list of .safetensors files.
    """
    if not os.path.exists(lora_folder):
        return ["None"]

    lora_items = []
    for item in os.listdir(lora_folder):
        if item.endswith(".safetensors"):
            lora_items.append(item)

    lora_items.sort(key=str.lower)
    return ["None"] + lora_items


def refresh_lora_dropdown(lora_folder: str):
    """Refresh LoRA dropdown choices."""
    new_choices = get_ltx_lora_options(lora_folder)
    return gr.update(choices=new_choices, value="None")


# =============================================================================
# Validation
# =============================================================================

def validate_num_frames(num_frames: int) -> Optional[str]:
    """Validate num_frames is in 8*K + 1 format."""
    if (num_frames - 1) % 8 != 0:
        valid_examples = [8*k + 1 for k in range(1, 20)]
        return f"Error: num_frames must be 8*K + 1 (e.g., {', '.join(map(str, valid_examples[:5]))}...)"
    return None


def validate_resolution(width: int, height: int) -> Optional[str]:
    """Validate resolution is divisible by 64."""
    if width % 64 != 0 or height % 64 != 0:
        return f"Error: width ({width}) and height ({height}) must be divisible by 64"
    return None


def validate_model_paths(checkpoint_path: str, spatial_upsampler_path: str,
                         distilled_lora_path: str, gemma_root: str,
                         is_one_stage: bool = False,
                         is_refine_only: bool = False) -> Optional[str]:
    """Validate all required model paths exist."""
    # Core paths required for all pipelines
    paths = [
        ("LTX Checkpoint", checkpoint_path),
        ("Gemma Root", gemma_root),
    ]
    # Two-stage specific paths (spatial upsampler always needed for two-stage)
    if not is_one_stage and not is_refine_only:
        paths.extend([
            ("Spatial Upsampler", spatial_upsampler_path),
            ("Distilled LoRA", distilled_lora_path),
        ])
    # Refine-only: distilled LoRA is optional, but validate if provided
    if is_refine_only:
        # Only validate distilled LoRA if path is provided
        if distilled_lora_path and distilled_lora_path.strip():
            paths.append(("Distilled LoRA", distilled_lora_path))
    for name, path in paths:
        if not path or not path.strip():
            return f"Error: {name} path is required"
        if not os.path.exists(path):
            return f"Error: {name} not found: {path}"
    return None


# =============================================================================
# Image Dimension Helpers
# =============================================================================

def update_image_dimensions(image_path):
    """Update original dimensions when image is uploaded."""
    if image_path is None:
        return "", gr.update(), gr.update()
    try:
        img = Image.open(image_path)
        w, h = img.size
        original_dims_str = f"{w}x{h}"
        # Calculate dimensions snapped to nearest multiple of 32 while maintaining aspect ratio
        new_w = round(w / 32) * 32
        new_h = round(h / 32) * 32
        new_w = max(32, new_w)
        new_h = max(32, new_h)
        return original_dims_str, gr.update(value=new_w), gr.update(value=new_h)
    except Exception as e:
        print(f"Error reading image dimensions: {e}")
        return "", gr.update(), gr.update()


def update_resolution_from_scale(scale, original_dims):
    """Update dimensions based on scale percentage (divisible by 64)."""
    if not original_dims:
        return gr.update(), gr.update()
    try:
        scale = float(scale) if scale is not None else 100.0
        if scale <= 0:
            scale = 100.0

        orig_w, orig_h = map(int, original_dims.split('x'))
        scale_factor = scale / 100.0

        # Calculate and round to the nearest multiple of 64
        new_w = round((orig_w * scale_factor) / 64) * 64
        new_h = round((orig_h * scale_factor) / 64) * 64

        # Ensure minimum size (must be multiple of 64)
        new_w = max(64, new_w)
        new_h = max(64, new_h)

        return gr.update(value=new_w), gr.update(value=new_h)
    except Exception as e:
        print(f"Error updating from scale: {e}")
        return gr.update(), gr.update()


def calculate_width_from_height(height, original_dims):
    """Calculate width based on height maintaining aspect ratio (divisible by 64)."""
    if not original_dims or height is None:
        return gr.update()
    try:
        height = int(height)
        if height <= 0:
            return gr.update()
        height = (height // 64) * 64
        height = max(64, height)

        orig_w, orig_h = map(int, original_dims.split('x'))
        if orig_h == 0:
            return gr.update()
        aspect_ratio = orig_w / orig_h
        new_width = round((height * aspect_ratio) / 64) * 64
        return gr.update(value=max(64, new_width))

    except Exception as e:
        print(f"Error calculating width: {e}")
        return gr.update()


def calculate_height_from_width(width, original_dims):
    """Calculate height based on width maintaining aspect ratio (divisible by 64)."""
    if not original_dims or width is None:
        return gr.update()
    try:
        width = int(width)
        if width <= 0:
            return gr.update()
        width = (width // 64) * 64
        width = max(64, width)

        orig_w, orig_h = map(int, original_dims.split('x'))
        if orig_w == 0:
            return gr.update()
        aspect_ratio = orig_w / orig_h
        new_height = round((width / aspect_ratio) / 64) * 64
        return gr.update(value=max(64, new_height))

    except Exception as e:
        print(f"Error calculating height: {e}")
        return gr.update()


def get_video_info(video_path: str) -> dict:
    """Get video information using PyAV."""
    try:
        import av
        with av.open(video_path) as container:
            # Get first video stream
            video_stream = container.streams.video[0]

            width = video_stream.width
            height = video_stream.height

            # Calculate FPS from average_rate or guessed_rate
            if video_stream.average_rate:
                fps = float(video_stream.average_rate)
            elif video_stream.guessed_rate:
                fps = float(video_stream.guessed_rate)
            else:
                fps = 30.0  # fallback

            # Calculate duration
            if video_stream.duration and video_stream.time_base:
                duration = float(video_stream.duration * video_stream.time_base)
            elif container.duration:
                duration = container.duration / av.time_base
            else:
                duration = 0

            total_frames = int(video_stream.frames) if video_stream.frames else int(duration * fps)

            return {
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames,
                'duration': duration
            }
    except Exception as e:
        print(f"Error extracting video info: {e}")
        return {}


def update_depth_control_status(video_path: str, image_path: str) -> str:
    """Get status info for depth control video or image."""
    if video_path and os.path.exists(video_path):
        info = get_video_info(video_path)
        if info:
            w = info.get("width", 0)
            h = info.get("height", 0)
            fps = info.get("fps", 0)
            frames = info.get("total_frames", 0)
            return f"Video: {w}x{h} | {fps:.2f} FPS | {frames} frames"
        return "Video: Unable to read info"
    elif image_path and os.path.exists(image_path):
        try:
            img = Image.open(image_path)
            w, h = img.size
            return f"Image: {w}x{h} | N/A FPS | 1 frame"
        except Exception:
            return "Image: Unable to read info"
    return "No depth map loaded"


def update_video_dimensions(video_path):
    """Update dimensions and frame count when video is uploaded."""
    if video_path is None:
        return "", gr.update(), gr.update(), gr.update(), gr.update()
    try:
        info = get_video_info(video_path)
        if info:
            w, h = info.get("width", 0), info.get("height", 0)
            if w and h:
                original_dims_str = f"{w}x{h}"
                # Snap to nearest multiple of 32
                new_w = round(w / 32) * 32
                new_h = round(h / 32) * 32
                new_w = max(32, new_w)
                new_h = max(32, new_h)
                # Calculate frame count (snap to 8*K+1 format)
                if info.get("total_frames"):
                    num_frames = info["total_frames"]
                elif info.get("duration") and info.get("fps"):
                    num_frames = int(info["duration"] * info["fps"])
                else:
                    num_frames = 121
                # Snap to nearest 8*K+1
                k = max(1, round((num_frames - 1) / 8))
                num_frames = 8 * k + 1
                fps = info.get("fps", 24)
                return original_dims_str, gr.update(value=new_w), gr.update(value=new_h), gr.update(value=num_frames), gr.update(value=int(fps))
    except Exception as e:
        print(f"Error reading video dimensions: {e}")
    return "", gr.update(), gr.update(), gr.update(), gr.update()


def extract_video_metadata(video_path: str) -> dict:
    """Extract metadata from video file using PyAV."""
    try:
        import av
        with av.open(video_path) as container:
            # Get comment metadata from container
            comment = container.metadata.get('comment', '{}')
            return json.loads(comment)
    except Exception as e:
        print(f"Metadata extraction failed: {str(e)}")
        return {}


def add_metadata_to_video(video_path: str, parameters: dict) -> None:
    """Add generation parameters to video metadata using ffmpeg."""
    # Convert parameters to JSON string
    params_json = json.dumps(parameters, indent=2)

    # Temporary output path
    temp_path = video_path.replace(".mp4", "_temp.mp4")

    # FFmpeg command to add metadata without re-encoding
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-metadata', f'comment={params_json}',
        '-codec', 'copy',
        temp_path
    ]

    try:
        # Execute FFmpeg command
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Replace original file with the metadata-enhanced version
        os.replace(temp_path, video_path)
    except subprocess.CalledProcessError as e:
        print(f"Failed to add metadata: {e.stderr.decode()}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as e:
        print(f"Error: {str(e)}")


def extract_first_frame(video_path: str, output_dir: str = "outputs") -> Optional[str]:
    """Extract the first frame from a video file."""
    try:
        import av
        os.makedirs(output_dir, exist_ok=True)

        with av.open(video_path) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                # Convert to PIL Image and save
                img = frame.to_image()
                # Save with unique name based on source video
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}_frame0.png")
                img.save(output_path)
                return output_path
    except Exception as e:
        print(f"Error extracting first frame: {e}")
    return None


def extract_video_details(video_path: str) -> Tuple[dict, str, Optional[str]]:
    """Extract metadata, video information, and first frame."""
    metadata = extract_video_metadata(video_path)
    video_details = get_video_info(video_path)

    # Combine metadata with video details
    for key, value in video_details.items():
        if key not in metadata:
            metadata[key] = value

    # Extract first frame
    first_frame_path = extract_first_frame(video_path)

    # Return metadata, status message, and first frame path
    return metadata, "Video details extracted successfully", first_frame_path


# =============================================================================
# GIMM-VFI Frame Interpolation
# =============================================================================

def _gimm_set_seed(seed=None):
    """Set random seed for reproducibility (inlined from GIMM-VFI)."""
    import random
    import numpy as np
    import torch
    if seed is None:
        seed = random.getrandbits(32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


class _GIMMInputPadder:
    """Pads images such that dimensions are divisible by divisor (inlined from GIMM-VFI)."""

    def __init__(self, dims, divisor=16):
        import torch.nn.functional as F
        self.F = F
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        self._pad = [
            pad_wd // 2,
            pad_wd - pad_wd // 2,
            pad_ht // 2,
            pad_ht - pad_ht // 2,
        ]

    def pad(self, *inputs):
        if len(inputs) == 1:
            return self.F.pad(inputs[0], self._pad, mode="replicate")
        else:
            return [self.F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, *inputs):
        if len(inputs) == 1:
            return self._unpad(inputs[0])
        else:
            return [self._unpad(x) for x in inputs]

    def _unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


# Model variant configurations
GIMM_MODELS = {
    "GIMM-VFI-R (RAFT)": {
        "type": "gimm",
        "config": "GIMM-VFI/configs/gimmvfi/gimmvfi_r_arb.yaml",
        "checkpoint": "GIMM-VFI/pretrained_ckpt/gimmvfi_r_arb.pt",
    },
    "GIMM-VFI-R-P (RAFT+Perceptual)": {
        "type": "gimm",
        "config": "GIMM-VFI/configs/gimmvfi/gimmvfi_r_arb.yaml",
        "checkpoint": "GIMM-VFI/pretrained_ckpt/gimmvfi_r_arb_lpips.pt",
    },
    "GIMM-VFI-F (FlowFormer)": {
        "type": "gimm",
        "config": "GIMM-VFI/configs/gimmvfi/gimmvfi_f_arb.yaml",
        "checkpoint": "GIMM-VFI/pretrained_ckpt/gimmvfi_f_arb.pt",
    },
    "GIMM-VFI-F-P (FlowFormer+Perceptual)": {
        "type": "gimm",
        "config": "GIMM-VFI/configs/gimmvfi/gimmvfi_f_arb.yaml",
        "checkpoint": "GIMM-VFI/pretrained_ckpt/gimmvfi_f_arb_lpips.pt",
    },
    "BiM-VFI (Bidirectional Motion)": {
        "type": "bim",
        "checkpoint": "GIMM-VFI/pretrained_ckpt/bim_vfi.pth",
    },
}

# Upscaler model configurations
UPSCALER_MODELS = {
    "Real-ESRGAN x2": {
        "type": "esrgan",
        "scale": 2,
        "checkpoint": "GIMM-VFI/pretrained_ckpt/RealESRGAN_x2plus.pth",
    },
    "Real-ESRGAN x4": {
        "type": "esrgan",
        "scale": 4,
        "checkpoint": "GIMM-VFI/pretrained_ckpt/RealESRGAN_x4plus.pth",
    },
    "SwinIR x4": {
        "type": "swinir",
        "scale": 4,
        "checkpoint": "GIMM-VFI/pretrained_ckpt/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
    },
    "BasicVSR++ x4 (Temporal)": {
        "type": "basicvsr",
        "scale": 4,
        "checkpoint": "GIMM-VFI/pretrained_ckpt/basicvsr_plusplus_reds4.pth",
    },
}

# Global cache for GIMM-VFI model
_gimm_model_cache = {"model": None, "variant": None}

# Global cache for BiM-VFI model
_bim_model_cache = {"model": None, "variant": None}


def _load_gimm_model(model_variant: str, checkpoint_path: str = "", config_path: str = ""):
    """Load GIMM-VFI model with caching."""
    import torch

    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gimm_dir = os.path.join(script_dir, "GIMM-VFI")
    gimm_src_path = os.path.join(gimm_dir, "src")

    # Add GIMM-VFI to path
    if gimm_src_path not in sys.path:
        sys.path.insert(0, gimm_src_path)

    from models import create_model
    from utils.setup import single_setup

    # Use defaults if not specified
    model_info = GIMM_MODELS.get(model_variant, GIMM_MODELS["GIMM-VFI-R-P (RAFT+Perceptual)"])
    config_file = config_path if config_path else model_info["config"]
    ckpt_file = checkpoint_path if checkpoint_path else model_info["checkpoint"]

    # Check if already loaded
    cache_key = f"{model_variant}:{ckpt_file}"
    if _gimm_model_cache["model"] is not None and _gimm_model_cache["variant"] == cache_key:
        return _gimm_model_cache["model"]

    # Clear old model from GPU
    if _gimm_model_cache["model"] is not None:
        del _gimm_model_cache["model"]
        _gimm_model_cache["model"] = None
        torch.cuda.empty_cache()

    # Setup config via argparse.Namespace (required by GIMM-VFI single_setup)
    # Use absolute path for config file
    import argparse
    abs_config_file = os.path.join(script_dir, config_file) if not os.path.isabs(config_file) else config_file
    args = argparse.Namespace(
        eval=True,
        resume=False,
        seed=0,
        model_config=abs_config_file,
    )
    config = single_setup(args, extra_args=[])

    # Create model - need to change to GIMM-VFI dir because RAFT uses relative paths
    original_cwd = os.getcwd()
    try:
        os.chdir(gimm_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = create_model(config.arch)
        model = model.to(device)
    finally:
        os.chdir(original_cwd)

    # Load checkpoint (use absolute path)
    abs_ckpt_file = os.path.join(script_dir, ckpt_file) if not os.path.isabs(ckpt_file) else ckpt_file
    if os.path.exists(abs_ckpt_file):
        ckpt = torch.load(abs_ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {abs_ckpt_file}")

    model.eval()

    # Cache the model
    _gimm_model_cache["model"] = model
    _gimm_model_cache["variant"] = cache_key

    return model


def _load_bim_model(checkpoint_path: str = ""):
    """Load BiM-VFI model with caching."""
    import torch

    script_dir = os.path.dirname(os.path.abspath(__file__))
    gimm_dir = os.path.join(script_dir, "GIMM-VFI")

    # Add GIMM-VFI to path so bim_vfi can be imported as a package
    if gimm_dir not in sys.path:
        sys.path.insert(0, gimm_dir)

    from bim_vfi import BiMVFI

    # Use default checkpoint if not specified
    default_ckpt = "GIMM-VFI/pretrained_ckpt/bim_vfi.pth"
    ckpt_file = checkpoint_path if checkpoint_path else default_ckpt

    # Check if already loaded
    cache_key = f"bim:{ckpt_file}"
    if _bim_model_cache["model"] is not None and _bim_model_cache["variant"] == cache_key:
        return _bim_model_cache["model"]

    # Clear old model from GPU
    if _bim_model_cache["model"] is not None:
        del _bim_model_cache["model"]
        _bim_model_cache["model"] = None
        torch.cuda.empty_cache()

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiMVFI(pyr_level=3, feat_channels=32)
    model = model.to(device)

    # Load checkpoint (use absolute path)
    abs_ckpt_file = os.path.join(script_dir, ckpt_file) if not os.path.isabs(ckpt_file) else ckpt_file
    if os.path.exists(abs_ckpt_file):
        ckpt = torch.load(abs_ckpt_file, map_location="cpu", weights_only=False)
        # Support "model", "state_dict" keys or direct state dict
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        model.load_state_dict(state_dict, strict=True)
    else:
        raise FileNotFoundError(f"BiM-VFI checkpoint not found: {abs_ckpt_file}")

    model.eval()

    # Cache the model
    _bim_model_cache["model"] = model
    _bim_model_cache["variant"] = cache_key

    return model


def _extract_video_frames(video_path: str, output_dir: str) -> Tuple[List[str], float]:
    """Extract frames from video using ffmpeg. Returns frame paths and FPS."""
    os.makedirs(output_dir, exist_ok=True)

    # Get video FPS
    probe_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    fps_str = result.stdout.strip()
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den)
    else:
        fps = float(fps_str) if fps_str else 24.0

    # Extract frames
    extract_cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-qscale:v", "2",
        os.path.join(output_dir, "%05d.png")
    ]
    subprocess.run(extract_cmd, capture_output=True)

    # Get sorted frame paths
    frame_paths = sorted(Path(output_dir).glob("*.png"))
    return [str(p) for p in frame_paths], fps


def _frames_to_video(frame_dir: str, output_path: str, fps: float, audio_source: str = None):
    """Reassemble frames into video using ffmpeg, optionally adding audio."""
    if audio_source:
        # Create video without audio first
        temp_video = output_path + ".temp.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(frame_dir, "%05d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            temp_video
        ]
        subprocess.run(cmd, capture_output=True)

        # Mux audio from source (stretch/compress to match new duration)
        # Get durations
        probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", temp_video]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        new_duration = float(result.stdout.strip()) if result.stdout.strip() else 0

        probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", audio_source]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        orig_duration = float(result.stdout.strip()) if result.stdout.strip() else 0

        if orig_duration > 0 and new_duration > 0:
            # Calculate tempo adjustment for audio to match video duration
            tempo = orig_duration / new_duration
            # atempo filter only accepts 0.5 to 2.0, chain multiple if needed
            atempo_filters = []
            t = tempo
            while t > 2.0:
                atempo_filters.append("atempo=2.0")
                t /= 2.0
            while t < 0.5:
                atempo_filters.append("atempo=0.5")
                t *= 2.0
            atempo_filters.append(f"atempo={t}")
            atempo_chain = ",".join(atempo_filters)

            cmd = [
                "ffmpeg", "-y",
                "-i", temp_video,
                "-i", audio_source,
                "-c:v", "copy",
                "-af", atempo_chain,
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-shortest",
                output_path
            ]
        else:
            # Just copy audio without tempo adjustment
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_video,
                "-i", audio_source,
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-shortest",
                output_path
            ]

        subprocess.run(cmd, capture_output=True)
        # Clean up temp file
        try:
            os.remove(temp_video)
        except:
            pass
    else:
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(frame_dir, "%05d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            output_path
        ]
        subprocess.run(cmd, capture_output=True)


def interpolate_video_gimm(
    input_video: str,
    model_variant: str,
    checkpoint_path: str,
    config_path: str,
    interp_factor: int,
    ds_scale: float,
    output_fps_override: float,
    raft_iters: int,
    seed: int,
) -> Generator[Tuple[Optional[str], str, float], None, None]:
    """
    Interpolate video frames using GIMM-VFI.

    Yields: (output_video_path, status_text, progress_fraction)
    """
    import torch
    import numpy as np
    import cv2
    import gc

    if not input_video:
        yield None, "Error: No input video provided", 0.0
        return

    _gimm_set_seed(seed)

    # Create temp directories
    temp_dir = tempfile.mkdtemp(prefix="gimm_interp_")
    input_frames_dir = os.path.join(temp_dir, "input_frames")
    output_frames_dir = os.path.join(temp_dir, "output_frames")
    os.makedirs(input_frames_dir, exist_ok=True)
    os.makedirs(output_frames_dir, exist_ok=True)

    try:
        # Extract frames
        yield None, "Extracting video frames...", 0.05
        frame_paths, original_fps = _extract_video_frames(input_video, input_frames_dir)

        if len(frame_paths) < 2:
            yield None, "Error: Video must have at least 2 frames", 0.0
            return

        yield None, f"Extracted {len(frame_paths)} frames at {original_fps:.2f} FPS", 0.1

        # Load model
        yield None, f"Loading {model_variant}...", 0.15
        model = _load_gimm_model(model_variant, checkpoint_path, config_path)
        device = next(model.parameters()).device

        yield None, "Model loaded, starting interpolation...", 0.2

        # Process frame pairs
        N = interp_factor  # Number of output frames per input pair (including endpoints)
        total_pairs = len(frame_paths) - 1
        output_frame_idx = 0

        def load_image(img_path):
            from PIL import Image
            img = Image.open(img_path)
            raw_img = np.array(img.convert("RGB"))
            img_tensor = torch.from_numpy(raw_img.copy()).permute(2, 0, 1) / 255.0
            return img_tensor.to(torch.float).unsqueeze(0)

        for pair_idx in range(total_pairs):
            progress = 0.2 + (pair_idx / total_pairs) * 0.7
            yield None, f"Interpolating pair {pair_idx + 1}/{total_pairs}...", progress

            # Load frame pair
            I0 = load_image(frame_paths[pair_idx]).to(device)
            I2 = load_image(frame_paths[pair_idx + 1]).to(device)

            # Pad to divisible by 32
            padder = _GIMMInputPadder(I0.shape, 32)
            I0_pad, I2_pad = padder.pad(I0, I2)

            # Create batch
            xs = torch.cat((I0_pad.unsqueeze(2), I2_pad.unsqueeze(2)), dim=2)
            batch_size = xs.shape[0]
            s_shape = xs.shape[-2:]

            # Save first frame (only for first pair)
            if pair_idx == 0:
                frame_np = (I0[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_frames_dir, f"{output_frame_idx:05d}.png"), frame_bgr)
                output_frame_idx += 1

            # Generate intermediate frames
            with torch.no_grad():
                coord_inputs = [
                    (
                        model.sample_coord_input(
                            batch_size,
                            s_shape,
                            [1 / N * i],
                            device=xs.device,
                            upsample_ratio=ds_scale,
                        ),
                        None,
                    )
                    for i in range(1, N)
                ]
                timesteps = [
                    i * 1 / N * torch.ones(batch_size).to(xs.device).to(torch.float)
                    for i in range(1, N)
                ]

                outputs = model(xs, coord_inputs, t=timesteps, ds_factor=ds_scale)
                out_frames = [padder.unpad(im) for im in outputs["imgt_pred"]]

            # Save interpolated frames
            for frame_tensor in out_frames:
                frame_np = (frame_tensor[0].cpu().numpy().transpose(1, 2, 0) * 255.0)
                frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_frames_dir, f"{output_frame_idx:05d}.png"), frame_bgr)
                output_frame_idx += 1

            # Save second frame of pair
            frame_np = (padder.unpad(I2_pad)[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_frames_dir, f"{output_frame_idx:05d}.png"), frame_bgr)
            output_frame_idx += 1

            # Clear CUDA cache periodically
            if pair_idx % 10 == 0:
                torch.cuda.empty_cache()

        # Reassemble video
        yield None, "Encoding output video...", 0.92

        # Calculate output FPS
        output_fps = output_fps_override if output_fps_override > 0 else original_fps * N

        # Create output path
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"interpolated_{int(time.time())}.mp4"
        output_path = os.path.join(output_dir, output_filename)

        _frames_to_video(output_frames_dir, output_path, output_fps, audio_source=input_video)

        # Offload model and clear VRAM before final yield
        if _gimm_model_cache["model"] is not None:
            _gimm_model_cache["model"].cpu()
            del _gimm_model_cache["model"]
            _gimm_model_cache["model"] = None
            _gimm_model_cache["variant"] = None
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        yield output_path, f"Done! Output: {output_fps:.1f} FPS ({output_frame_idx} frames)", 1.0

    except Exception as e:
        yield None, f"Error: {str(e)}", 0.0
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup temp directory
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        # Ensure model is offloaded even on error
        try:
            if _gimm_model_cache["model"] is not None:
                _gimm_model_cache["model"].cpu()
                del _gimm_model_cache["model"]
                _gimm_model_cache["model"] = None
                _gimm_model_cache["variant"] = None
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass


def interpolate_video_bim(
    input_video: str,
    checkpoint_path: str,
    interp_factor: int,
    pyr_level: int,
    output_fps_override: float,
    seed: int,
) -> Generator[Tuple[Optional[str], str, float], None, None]:
    """
    Interpolate video frames using BiM-VFI.

    Yields: (output_video_path, status_text, progress_fraction)
    """
    import torch
    import numpy as np
    import cv2
    import gc

    if not input_video:
        yield None, "Error: No input video provided", 0.0
        return

    _gimm_set_seed(seed)

    # Create temp directories
    temp_dir = tempfile.mkdtemp(prefix="bim_interp_")
    input_frames_dir = os.path.join(temp_dir, "input_frames")
    output_frames_dir = os.path.join(temp_dir, "output_frames")
    os.makedirs(input_frames_dir, exist_ok=True)
    os.makedirs(output_frames_dir, exist_ok=True)

    try:
        # Extract frames
        yield None, "Extracting video frames...", 0.05
        frame_paths, original_fps = _extract_video_frames(input_video, input_frames_dir)

        if len(frame_paths) < 2:
            yield None, "Error: Video must have at least 2 frames", 0.0
            return

        yield None, f"Extracted {len(frame_paths)} frames at {original_fps:.2f} FPS", 0.1

        # Load model
        yield None, "Loading BiM-VFI model...", 0.15
        model = _load_bim_model(checkpoint_path)
        device = next(model.parameters()).device

        yield None, "Model loaded, starting interpolation...", 0.2

        # Auto-detect pyr_level based on resolution if not specified
        from PIL import Image
        first_img = Image.open(frame_paths[0])
        width, height = first_img.size
        max_dim = max(width, height)

        if pyr_level <= 0:
            # Auto-detect based on resolution
            if max_dim >= 3840:  # 4K+
                auto_pyr_level = 7
            elif max_dim >= 1920:  # 1080p
                auto_pyr_level = 6
            else:  # < 1080p
                auto_pyr_level = 5
            yield None, f"Auto-detected pyr_level={auto_pyr_level} for {width}x{height}", 0.22
        else:
            auto_pyr_level = pyr_level

        # Process frame pairs
        N = interp_factor  # Number of output frames per input pair (including endpoints)
        total_pairs = len(frame_paths) - 1
        output_frame_idx = 0

        def load_image(img_path):
            img = Image.open(img_path)
            raw_img = np.array(img.convert("RGB"))
            img_tensor = torch.from_numpy(raw_img.copy()).permute(2, 0, 1) / 255.0
            return img_tensor.to(torch.float).unsqueeze(0)

        for pair_idx in range(total_pairs):
            progress = 0.2 + (pair_idx / total_pairs) * 0.7
            yield None, f"Interpolating pair {pair_idx + 1}/{total_pairs}...", progress

            # Load frame pair
            I0 = load_image(frame_paths[pair_idx]).to(device)
            I1 = load_image(frame_paths[pair_idx + 1]).to(device)

            # Save first frame (only for first pair)
            if pair_idx == 0:
                frame_np = (I0[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_frames_dir, f"{output_frame_idx:05d}.png"), frame_bgr)
                output_frame_idx += 1

            # Generate intermediate frames
            with torch.no_grad():
                for i in range(1, N):
                    time_step = i / N
                    results = model(img0=I0, img1=I1, time_step=time_step, pyr_level=auto_pyr_level)
                    imgt_pred = results["imgt_pred"]

                    # Save interpolated frame
                    frame_np = (imgt_pred[0].cpu().numpy().transpose(1, 2, 0) * 255.0)
                    frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
                    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(output_frames_dir, f"{output_frame_idx:05d}.png"), frame_bgr)
                    output_frame_idx += 1

            # Save second frame of pair
            frame_np = (I1[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_frames_dir, f"{output_frame_idx:05d}.png"), frame_bgr)
            output_frame_idx += 1

            # Clear CUDA cache periodically
            if pair_idx % 10 == 0:
                torch.cuda.empty_cache()

        # Reassemble video
        yield None, "Encoding output video...", 0.92

        # Calculate output FPS
        output_fps = output_fps_override if output_fps_override > 0 else original_fps * N

        # Create output path
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"interpolated_bim_{int(time.time())}.mp4"
        output_path = os.path.join(output_dir, output_filename)

        _frames_to_video(output_frames_dir, output_path, output_fps, audio_source=input_video)

        # Offload model and clear VRAM before final yield
        if _bim_model_cache["model"] is not None:
            _bim_model_cache["model"].cpu()
            del _bim_model_cache["model"]
            _bim_model_cache["model"] = None
            _bim_model_cache["variant"] = None
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        yield output_path, f"Done! Output: {output_fps:.1f} FPS ({output_frame_idx} frames)", 1.0

    except Exception as e:
        yield None, f"Error: {str(e)}", 0.0
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup temp directory
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        # Ensure model is offloaded even on error
        try:
            if _bim_model_cache["model"] is not None:
                _bim_model_cache["model"].cpu()
                del _bim_model_cache["model"]
                _bim_model_cache["model"] = None
                _bim_model_cache["variant"] = None
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass


def interpolate_video(
    input_video: str,
    model_variant: str,
    checkpoint_path: str,
    config_path: str,
    interp_factor: int,
    ds_scale: float,
    output_fps_override: float,
    raft_iters: int,
    pyr_level: int,
    seed: int,
) -> Generator[Tuple[Optional[str], str, float], None, None]:
    """
    Unified dispatcher for video frame interpolation.
    Runs interpolation in a subprocess for complete VRAM cleanup.
    """
    if not input_video:
        yield None, "Error: No input video provided", 0.0
        return

    model_info = GIMM_MODELS.get(model_variant, {})
    model_type = model_info.get("type", "gimm")

    # Create output path
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"interpolated_{model_type}_{int(time.time())}.mp4"
    output_path = os.path.join(output_dir, output_filename)

    # Build subprocess command
    script_dir = os.path.dirname(os.path.abspath(__file__))
    interp_script = os.path.join(script_dir, "interpolate_video.py")

    command = [
        sys.executable, interp_script,
        "--input", input_video,
        "--output", output_path,
        "--model-type", model_type,
        "--variant", model_variant,
        "--factor", str(int(interp_factor)),
        "--pyr-level", str(int(pyr_level)),
        "--ds-scale", str(float(ds_scale)),
        "--output-fps", str(float(output_fps_override)),
        "--seed", str(int(seed)),
    ]

    if checkpoint_path:
        command.extend(["--checkpoint", checkpoint_path])
    if config_path:
        command.extend(["--config", config_path])

    print("\n" + "=" * 80)
    print("LAUNCHING INTERPOLATION SUBPROCESS:")
    print(" ".join(command))
    print("=" * 80 + "\n")

    yield None, "Starting interpolation subprocess...", 0.05

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )

        output_file = None
        last_status = "Processing..."

        while True:
            line = process.stdout.readline()
            if line:
                line = line.strip()
                if line:
                    print(line)
                    if line.startswith("PROGRESS:"):
                        last_status = line[9:].strip()
                        # Try to extract percentage
                        progress = 0.1
                        if "%" in last_status:
                            try:
                                pct = int(last_status.split("(")[1].split("%")[0])
                                progress = 0.1 + (pct / 100) * 0.8
                            except:
                                pass
                        yield None, last_status, progress
                    elif line.startswith("OUTPUT:"):
                        output_file = line[7:].strip()
                    elif line.startswith("ERROR:"):
                        yield None, line[6:].strip(), 0.0
                        return

            if process.poll() is not None:
                # Read remaining output
                for line in process.stdout:
                    line = line.strip()
                    if line:
                        print(line)
                        if line.startswith("OUTPUT:"):
                            output_file = line[7:].strip()
                        elif line.startswith("ERROR:"):
                            yield None, line[6:].strip(), 0.0
                            return
                break

        return_code = process.returncode

        if return_code == 0 and output_file and os.path.exists(output_file):
            yield output_file, f"Done! Output saved to {output_file}", 1.0
        else:
            yield None, f"Interpolation failed (exit code {return_code})", 0.0

    except Exception as e:
        yield None, f"Error: {str(e)}", 0.0
        import traceback
        traceback.print_exc()


def upscale_video(
    input_video: str,
    model_variant: str,
    model_path_override: str,
    tile_size: int,
    half_precision: bool,
    motion_blur: bool,
    blur_strength: float,
    blur_samples: int,
    crf: int,
    seed: int,
) -> Generator[Tuple[Optional[str], str, float], None, None]:
    """
    Unified video upscaling dispatcher.
    Launches upscale_video.py as subprocess for VRAM cleanup.
    """
    # Validate input
    if not input_video or not os.path.exists(input_video):
        yield None, "Error: No input video provided", 0.0
        return

    # Get model config
    model_config = UPSCALER_MODELS.get(model_variant)
    if not model_config:
        yield None, f"Error: Unknown model variant: {model_variant}", 0.0
        return

    model_type = model_config["type"]
    scale = model_config["scale"]

    # Determine model path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if model_path_override and model_path_override.strip():
        model_path = model_path_override.strip()
    else:
        model_path = os.path.join(script_dir, model_config["checkpoint"])

    # Check if model exists
    if not os.path.exists(model_path):
        yield None, f"Error: Model not found at {model_path}", 0.0
        return

    # Create output path
    output_dir = os.path.join(script_dir, "outputs", "upscaled")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    input_name = os.path.splitext(os.path.basename(input_video))[0]
    output_path = os.path.join(output_dir, f"{input_name}_upscaled_{scale}x_{timestamp}.mp4")

    yield None, f"Starting {model_variant} upscaling...", 0.05

    # Build command
    cmd = [
        sys.executable, os.path.join(script_dir, "upscale_video.py"),
        "--input", input_video,
        "--output", output_path,
        "--model-type", model_type,
        "--model-path", model_path,
        "--scale", str(scale),
        "--tile-size", str(tile_size),
        "--crf", str(crf),
        "--seed", str(seed),
    ]

    if half_precision:
        cmd.append("--half")

    if motion_blur:
        cmd.extend([
            "--motion-blur",
            "--blur-strength", str(blur_strength),
            "--blur-samples", str(blur_samples),
        ])

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        output_file = None
        last_status = "Processing..."

        while True:
            line = process.stdout.readline()
            if line:
                line = line.strip()
                if line:
                    print(line)
                    if line.startswith("PROGRESS:"):
                        last_status = line[9:].strip()
                        # Try to extract percentage
                        progress = 0.1
                        if "%" in last_status:
                            try:
                                pct = int(last_status.split("(")[1].split("%")[0])
                                progress = 0.1 + (pct / 100) * 0.8
                            except:
                                pass
                        yield None, last_status, progress
                    elif line.startswith("OUTPUT:"):
                        output_file = line[7:].strip()
                    elif line.startswith("ERROR:"):
                        yield None, line[6:].strip(), 0.0
                        return

            if process.poll() is not None:
                # Read remaining output
                for line in process.stdout:
                    line = line.strip()
                    if line:
                        print(line)
                        if line.startswith("OUTPUT:"):
                            output_file = line[7:].strip()
                        elif line.startswith("ERROR:"):
                            yield None, line[6:].strip(), 0.0
                            return
                break

        return_code = process.returncode

        if return_code == 0 and output_file and os.path.exists(output_file):
            yield output_file, f"Done! Output saved to {output_file}", 1.0
        else:
            yield None, f"Upscaling failed (exit code {return_code})", 0.0

    except Exception as e:
        yield None, f"Error: {str(e)}", 0.0
        import traceback
        traceback.print_exc()


# =============================================================================
# Video Generation
# =============================================================================

def generate_ltx_video(
    # Prompts
    prompt: str,
    negative_prompt: str,
    # Model paths
    checkpoint_path: str,
    distilled_checkpoint: bool,
    stage2_checkpoint: str,
    gemma_root: str,
    spatial_upsampler_path: str,
    vae_path: str,
    distilled_lora_path: str,
    distilled_lora_strength: float,
    # Generation parameters
    mode: str,
    pipeline: str,
    sampler: str,
    stage2_sampler: str,
    enable_sliding_window: bool,
    width: int,
    height: int,
    num_frames: int,
    frame_rate: float,
    cfg_guidance_scale: float,
    num_inference_steps: int,
    stage2_steps: int,
    stage2_strength: float,
    stage3_strength: float,
    seed: int,
    # STG parameters
    stg_scale: float,
    stg_blocks: str,
    stg_mode: str,
    # Advanced CFG (MultiModal Guidance)
    guidance_mode: str,
    scheduler_type: str,
    video_cfg_guidance_scale: float,
    video_stg_scale: float,
    video_stg_blocks: str,
    video_rescale_scale: float,
    a2v_guidance_scale: float,
    video_skip_step: int,
    audio_cfg_guidance_scale: float,
    audio_stg_scale: float,
    audio_stg_blocks: str,
    audio_rescale_scale: float,
    v2a_guidance_scale: float,
    audio_skip_step: int,
    # Kandinsky scheduler parameters
    kandinsky_scheduler: bool,
    kandinsky_scheduler_scale: float,
    # Image conditioning (for I2V)
    input_image: str,
    image_frame_idx: int,
    image_strength: float,
    image_crf: float,
    # End image conditioning
    end_image: str,
    end_image_strength: float,
    end_image_crf: float,
    # Anchor image conditioning
    anchor_image: str,
    anchor_interval: int,
    anchor_strength: float,
    anchor_decay: str,
    anchor_crf: float,
    # Video input (for V2V / refine)
    input_video: str,
    refine_strength: float,
    # Audio & prompt
    disable_audio: bool,
    audio_input: str,
    audio_strength: float,
    v2v_audio_mode: str,
    v2v_audio_strength: float,
    enhance_prompt: bool,
    # Memory optimization
    offload: bool,
    enable_fp8: bool,
    enable_dit_block_swap: bool,
    dit_blocks_in_memory: int,
    enable_text_encoder_block_swap: bool,
    text_encoder_blocks_in_memory: int,
    enable_refiner_block_swap: bool,
    refiner_blocks_in_memory: int,
    ffn_chunk_size: int,
    enable_activation_offload: bool,
    temporal_chunk_size: int,
    # LoRA
    lora_folder: str,
    user_lora_1: str,
    user_lora_strength_1: float,
    user_lora_stage_1: str,
    user_lora_2: str,
    user_lora_strength_2: float,
    user_lora_stage_2: str,
    user_lora_3: str,
    user_lora_strength_3: float,
    user_lora_stage_3: str,
    user_lora_4: str,
    user_lora_strength_4: float,
    user_lora_stage_4: str,
    # Output
    save_path: str,
    batch_size: int,
    # Preview Generation
    enable_preview: bool,
    preview_interval: int,
    # Video Continuation (Frame Freezing)
    freeze_frames: int,
    freeze_transition: int,
    # Sliding Window (Long Video)
    sliding_window_size: int,
    sliding_window_overlap: int,
    sliding_window_overlap_noise: float,
    sliding_window_color_correction: float,
    # AV Extension (Time-Based Audio-Video Continuation)
    av_extend_video: str,
    av_extend_start_time: float,
    av_extend_end_time: float,
    av_extend_steps: int,
    av_extend_terminal: float,
    av_slope_len: int,
    av_no_stage2: bool,
    # Depth Control (IC-LoRA)
    depth_control_video: str,
    depth_control_image: str,
    estimate_depth: bool,
    depth_strength: float,
    depth_stage2: bool,
    # Latent Normalization (fixes overbaking and audio clipping)
    latent_norm_mode: str,
    latent_norm_factors: str,
    latent_norm_target_mean: float,
    latent_norm_target_std: float,
    latent_norm_percentile: float,
    latent_norm_clip_outliers: bool,
    latent_norm_video_only: bool,
    latent_norm_audio_only: bool,
    # V2A Mode (Video-to-Audio)
    v2a_mode: bool,
    v2a_strength: float,
    # Video Joining
    v2v_join_video1: str,
    v2v_join_video2: str,
    v2v_join_frames_check1: int,
    v2v_join_frames_check2: int,
    v2v_join_preserve1: float,
    v2v_join_preserve2: float,
    v2v_join_transition_time: float,
    v2v_join_steps: int,
    v2v_join_terminal: float,
) -> Generator[Tuple[List[Tuple[str, str]], Optional[str], str, str], None, None]:
    """
    Generate video using LTX-2 pipeline.

    Yields: (gallery_items, preview_path, status_text, progress_text)
    """
    global stop_event, current_process, current_output_filename
    stop_event.clear()
    current_process = None
    current_output_filename = None

    # Validate inputs
    error = validate_num_frames(int(num_frames))
    if error:
        yield [], None, error, ""
        return

    error = validate_resolution(int(width), int(height))
    if error:
        yield [], None, error, ""
        return

    # Round num_frames to valid 8*K+1 format
    num_frames = int(num_frames)
    k = max(1, round((num_frames - 1) / 8))
    num_frames = 8 * k + 1

    is_one_stage = (pipeline == "one-stage")
    is_refine_only = (pipeline == "refine-only")
    is_three_stage_exp = (pipeline == "three-stage-exp")
    is_three_stage_linear = (pipeline == "three-stage-linear")
    error = validate_model_paths(checkpoint_path, spatial_upsampler_path,
                                  distilled_lora_path, gemma_root,
                                  is_one_stage=is_one_stage or is_three_stage_linear,
                                  is_refine_only=is_refine_only)
    if error:
        yield [], None, error, ""
        return

    # Check image for I2V mode
    if mode == "i2v" and not input_image:
        yield [], None, "Error: Input image required for I2V mode", ""
        return

    # Check video for V2V/refine-only mode
    if (mode == "v2v" or pipeline == "refine-only") and not input_video:
        yield [], None, "Error: Input video required for V2V/refine-only mode", ""
        return

    os.makedirs(save_path, exist_ok=True)
    all_generated_videos = []

    for i in range(int(batch_size)):
        if stop_event.is_set():
            current_process = None
            yield all_generated_videos, None, "Generation stopped by user.", ""
            return

        # Seed handling
        current_seed = seed
        if seed == -1:
            current_seed = random.randint(0, 2**32 - 1)
        elif int(batch_size) > 1:
            current_seed = seed + i

        status_text = f"Processing {i+1}/{batch_size} (Seed: {current_seed})"
        yield all_generated_videos.copy(), None, status_text, "Starting generation..."

        timestamp = int(time.time())
        run_id = f"{timestamp}_{random.randint(1000, 9999)}"
        output_filename = os.path.join(save_path, f"ltx_{mode}_{timestamp}_{current_seed}.mp4")
        current_output_filename = output_filename

        # Build command
        command = [
            sys.executable, "ltx_generate_video.py",
            "--checkpoint-path", checkpoint_path,
            "--gemma-root", gemma_root,
            "--prompt", str(prompt),
            "--negative-prompt", str(negative_prompt),
            "--num-frames", str(int(num_frames)),
            "--frame-rate", str(float(frame_rate)),
            "--width", str(int(width)),
            "--height", str(int(height)),
            "--cfg-guidance-scale", str(float(cfg_guidance_scale)),
            "--num-inference-steps", str(int(num_inference_steps)),
            "--stage2-steps", str(int(stage2_steps)),
            "--stage2-strength", str(float(stage2_strength)),
            "--stage3-strength", str(float(stage3_strength)),
            "--seed", str(current_seed),
            "--output-path", output_filename,
            "--sampler", str(sampler),
            "--stage2-sampler", str(stage2_sampler),
        ]

        # STG parameters (only include when stg_scale > 0)
        if float(stg_scale) > 0:
            command.extend(["--stg-scale", str(float(stg_scale))])
            command.extend(["--stg-mode", str(stg_mode)])
            # STG blocks (parse comma-separated string to list)
            if stg_blocks and stg_blocks.strip():
                for block in stg_blocks.split(","):
                    block = block.strip()
                    if block:
                        command.extend(["--stg-blocks", block])

        # Advanced CFG (MultiModal Guidance)
        if guidance_mode and guidance_mode != "legacy":
            command.extend(["--guidance-mode", str(guidance_mode)])
            command.extend(["--video-cfg-guidance-scale", str(float(video_cfg_guidance_scale))])
            command.extend(["--video-stg-scale", str(float(video_stg_scale))])
            command.extend(["--video-rescale-scale", str(float(video_rescale_scale))])
            command.extend(["--a2v-guidance-scale", str(float(a2v_guidance_scale))])
            command.extend(["--video-skip-step", str(int(video_skip_step))])
            command.extend(["--audio-cfg-guidance-scale", str(float(audio_cfg_guidance_scale))])
            command.extend(["--audio-stg-scale", str(float(audio_stg_scale))])
            command.extend(["--audio-rescale-scale", str(float(audio_rescale_scale))])
            command.extend(["--v2a-guidance-scale", str(float(v2a_guidance_scale))])
            command.extend(["--audio-skip-step", str(int(audio_skip_step))])
            # Video STG blocks (multimodal)
            if video_stg_blocks and video_stg_blocks.strip():
                for block in video_stg_blocks.split(","):
                    block = block.strip()
                    if block:
                        command.extend(["--video-stg-blocks", block])
            # Audio STG blocks (multimodal)
            if audio_stg_blocks and audio_stg_blocks.strip():
                for block in audio_stg_blocks.split(","):
                    block = block.strip()
                    if block:
                        command.extend(["--audio-stg-blocks", block])

        # Scheduler type
        if scheduler_type and scheduler_type != "ltx2":
            command.extend(["--scheduler", str(scheduler_type)])

        # Kandinsky scheduler parameters
        if kandinsky_scheduler:
            command.append("--kandinsky-scheduler")
            command.extend(["--kandinsky-scheduler-scale", str(float(kandinsky_scheduler_scale))])

        # Pipeline selection
        if is_one_stage:
            command.append("--one-stage")
        elif is_refine_only:
            command.append("--refine-only")
            # Refine-only: distilled LoRA is optional, skip if stage2 checkpoint or distilled checkpoint
            if not distilled_checkpoint and not stage2_checkpoint and distilled_lora_path and distilled_lora_path.strip() and os.path.exists(distilled_lora_path):
                command.extend(["--distilled-lora", distilled_lora_path, str(distilled_lora_strength)])
        elif is_three_stage_exp:
            # Three-stage exponential: stage 1 at half res, stages 2 & 3 at full res
            command.extend(["--num-stages", "3", "--pipeline-mode", "exponential"])
            command.extend(["--spatial-upsampler-path", spatial_upsampler_path])
            if not distilled_checkpoint and not stage2_checkpoint:
                command.extend(["--distilled-lora", distilled_lora_path, str(distilled_lora_strength)])
        elif is_three_stage_linear:
            # Three-stage linear: all stages at full resolution
            command.extend(["--num-stages", "3", "--pipeline-mode", "linear"])
        else:
            # Two-stage specific: spatial upsampler and distilled LoRA
            command.extend(["--spatial-upsampler-path", spatial_upsampler_path])
            # Skip distilled LoRA if using stage2 checkpoint (full model) or distilled checkpoint
            if not distilled_checkpoint and not stage2_checkpoint:
                command.extend(["--distilled-lora", distilled_lora_path, str(distilled_lora_strength)])

        # VAE path (if provided)
        if vae_path and vae_path.strip() and os.path.exists(vae_path):
            command.extend(["--vae", vae_path])

        # Stage 2 checkpoint (full model for stage 2 refinement)
        if stage2_checkpoint and stage2_checkpoint.strip() and os.path.exists(stage2_checkpoint):
            command.extend(["--stage2-checkpoint", stage2_checkpoint])

        # Video input (V2V / refine)
        if input_video:
            command.extend(["--input-video", str(input_video)])
            command.extend(["--refine-strength", str(float(refine_strength))])

        # Image conditioning (I2V) - with per-image CRF
        if mode == "i2v" and input_image:
            command.extend(["--image", str(input_image), str(int(image_frame_idx)), str(float(image_strength)), str(int(image_crf))])

        # End image conditioning (place at last frame) - with per-image CRF
        if end_image:
            last_frame_idx = int(num_frames) - 1
            command.extend(["--image", str(end_image), str(last_frame_idx), str(float(end_image_strength)), str(int(end_image_crf))])

        # Anchor image conditioning (periodic guidance)
        if anchor_interval and int(anchor_interval) > 0:
            if anchor_image:
                command.extend(["--anchor-image", str(anchor_image)])
            command.extend(["--anchor-interval", str(int(anchor_interval))])
            command.extend(["--anchor-strength", str(float(anchor_strength))])
            if anchor_decay and anchor_decay != "none":
                command.extend(["--anchor-decay", str(anchor_decay)])
            if int(anchor_crf) != 33:  # Only add if non-default
                command.extend(["--anchor-crf", str(int(anchor_crf))])

        # Depth Control (IC-LoRA)
        if depth_control_video and os.path.exists(depth_control_video):
            command.extend(["--depth-video", str(depth_control_video)])
            command.extend(["--depth-strength", str(float(depth_strength))])
            if depth_stage2:
                command.append("--depth-stage2")
        elif depth_control_image and os.path.exists(depth_control_image):
            command.extend(["--depth-image", str(depth_control_image)])
            command.extend(["--depth-strength", str(float(depth_strength))])
            if depth_stage2:
                command.append("--depth-stage2")
        elif estimate_depth:
            command.append("--estimate-depth")
            command.extend(["--depth-strength", str(float(depth_strength))])
            if depth_stage2:
                command.append("--depth-stage2")

        # Latent Normalization (fixes overbaking and audio clipping)
        if latent_norm_mode and latent_norm_mode != "none":
            command.extend(["--latent-norm", str(latent_norm_mode)])
            command.extend(["--latent-norm-factors", str(latent_norm_factors)])
            command.extend(["--latent-norm-target-mean", str(float(latent_norm_target_mean))])
            command.extend(["--latent-norm-target-std", str(float(latent_norm_target_std))])
            command.extend(["--latent-norm-percentile", str(float(latent_norm_percentile))])
            if latent_norm_clip_outliers:
                command.append("--latent-norm-clip-outliers")
            if latent_norm_video_only:
                command.append("--latent-norm-video-only")
            if latent_norm_audio_only:
                command.append("--latent-norm-audio-only")

        # V2A Mode (Video-to-Audio)
        if v2a_mode:
            command.append("--v2a-mode")
            command.extend(["--v2a-strength", str(v2a_strength)])

        # Video Joining
        if v2v_join_video1 and v2v_join_video2:
            command.extend(["--v2v-join-video1", str(v2v_join_video1)])
            command.extend(["--v2v-join-video2", str(v2v_join_video2)])
            command.extend(["--v2v-join-frames-check1", str(int(v2v_join_frames_check1))])
            command.extend(["--v2v-join-frames-check2", str(int(v2v_join_frames_check2))])
            command.extend(["--v2v-join-preserve1", str(float(v2v_join_preserve1))])
            command.extend(["--v2v-join-preserve2", str(float(v2v_join_preserve2))])
            command.extend(["--v2v-join-transition-time", str(float(v2v_join_transition_time))])
            command.extend(["--v2v-join-steps", str(int(v2v_join_steps))])
            command.extend(["--v2v-join-terminal", str(float(v2v_join_terminal))])

        # User LoRAs - apply to selected stage(s)
        lora_configs = [
            (user_lora_1, user_lora_strength_1, user_lora_stage_1),
            (user_lora_2, user_lora_strength_2, user_lora_stage_2),
            (user_lora_3, user_lora_strength_3, user_lora_stage_3),
            (user_lora_4, user_lora_strength_4, user_lora_stage_4),
        ]
        for user_lora, user_lora_strength, user_lora_stage in lora_configs:
            if user_lora and user_lora != "None" and lora_folder:
                lora_path = os.path.join(lora_folder, user_lora)
                if os.path.exists(lora_path):
                    if user_lora_stage == "Stage 1 (Base)":
                        command.extend(["--lora", lora_path, str(user_lora_strength)])
                    elif user_lora_stage == "Stage 2 (Refine)":
                        command.extend(["--stage2-lora", lora_path, str(user_lora_strength)])
                    elif user_lora_stage == "Stage 3 (Refine)":
                        command.extend(["--stage3-lora", lora_path, str(user_lora_strength)])
                    else:  # "All"
                        command.extend(["--lora", lora_path, str(user_lora_strength)])
                        command.extend(["--stage2-lora", lora_path, str(user_lora_strength)])
                        command.extend(["--stage3-lora", lora_path, str(user_lora_strength)])

        # V2V Audio mode handling
        if input_video:
            command.extend(["--v2v-audio-mode", str(v2v_audio_mode)])
            if v2v_audio_mode == "condition":
                command.extend(["--v2v-audio-strength", str(float(v2v_audio_strength))])

        # Audio handling (external audio file)
        # Use external audio when: V2V mode is "external", or when not in V2V mode (T2V/I2V)
        if audio_input and os.path.exists(audio_input):
            if not input_video or v2v_audio_mode == "external":
                command.extend(["--audio", str(audio_input)])
                command.extend(["--audio-strength", str(float(audio_strength))])
        elif disable_audio:
            command.append("--disable-audio")

        # Flags
        if enhance_prompt:
            command.append("--enhance-prompt")
        if offload:
            command.append("--offload")
        if enable_fp8:
            command.append("--enable-fp8")
        if distilled_checkpoint:
            command.append("--distilled-checkpoint")
        # Block swap controls (separate for DiT, text encoder, and refiner)
        if enable_dit_block_swap:
            command.append("--enable-dit-block-swap")
            command.extend(["--dit-blocks-in-memory", str(int(dit_blocks_in_memory))])
        if enable_text_encoder_block_swap:
            command.append("--enable-text-encoder-block-swap")
            command.extend(["--text-encoder-blocks-in-memory", str(int(text_encoder_blocks_in_memory))])
        if enable_refiner_block_swap:
            command.append("--enable-refiner-block-swap")
            command.extend(["--refiner-blocks-in-memory", str(int(refiner_blocks_in_memory))])
        if ffn_chunk_size and int(ffn_chunk_size) > 0:
            command.extend(["--ffn-chunk-size", str(int(ffn_chunk_size))])
        if enable_activation_offload:
            command.append("--enable-activation-offload")
        if temporal_chunk_size and int(temporal_chunk_size) > 0:
            command.extend(["--temporal-chunk-size", str(int(temporal_chunk_size))])

        # Preview generation
        unique_preview_suffix = f"ltx_{run_id}"
        preview_mp4_path = None
        if enable_preview and preview_interval > 0:
            preview_base_dir = os.path.join(save_path, "previews")
            command.extend(["--preview-dir", preview_base_dir])
            command.extend(["--preview-interval", str(int(preview_interval))])
            command.extend(["--preview-suffix", unique_preview_suffix])
            preview_mp4_path = os.path.join(preview_base_dir, f"latent_preview_{unique_preview_suffix}.mp4")

        # Frame freezing (video continuation)
        if freeze_frames and int(freeze_frames) > 0:
            command.extend(["--freeze-frames", str(int(freeze_frames))])
            command.extend(["--freeze-transition", str(int(freeze_transition))])

        # Sliding window (long video) - requires explicit enable flag
        if enable_sliding_window:
            command.append("--enable-sliding-window")
            if sliding_window_size and int(sliding_window_size) > 0:
                command.extend(["--sliding-window-size", str(int(sliding_window_size))])
            command.extend(["--sliding-window-overlap", str(int(sliding_window_overlap))])
            if sliding_window_overlap_noise and float(sliding_window_overlap_noise) > 0:
                command.extend(["--sliding-window-overlap-noise", str(float(sliding_window_overlap_noise))])
            if sliding_window_color_correction and float(sliding_window_color_correction) > 0:
                command.extend(["--sliding-window-color-correction", str(float(sliding_window_color_correction))])

        # AV Extension (time-based audio-video continuation)
        if av_extend_video and os.path.exists(av_extend_video):
            command.extend(["--av-extend-from", str(av_extend_video)])
            if av_extend_start_time and float(av_extend_start_time) > 0:
                command.extend(["--av-extend-start-time", str(float(av_extend_start_time))])
            if av_extend_end_time and float(av_extend_end_time) > 0:
                command.extend(["--av-extend-end-time", str(float(av_extend_end_time))])
            command.extend(["--av-extend-steps", str(int(av_extend_steps))])
            command.extend(["--av-extend-terminal", str(float(av_extend_terminal))])
            command.extend(["--av-slope-len", str(int(av_slope_len))])
            if av_no_stage2:
                command.append("--av-no-stage2")

        # Print command for debugging
        print("\n" + "=" * 80)
        print(f"LAUNCHING COMMAND (Batch {i+1}/{batch_size}):")
        print(" ".join(command))
        print("=" * 80 + "\n")

        try:
            start_time = time.perf_counter()

            # Use PYTHONUNBUFFERED to ensure subprocess output isn't block-buffered
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env
            )

            current_process = process
            last_progress = ""
            current_preview_list = []
            last_preview_mtime = 0

            while True:
                if stop_event.is_set():
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    current_process = None
                    yield all_generated_videos, [], "Generation stopped by user.", ""
                    return

                # Read output line by line (blocking but with PYTHONUNBUFFERED)
                line = process.stdout.readline()
                if line:
                    line = line.strip()
                    if line:
                        print(line)
                        parsed = parse_ltx_progress_line(line)
                        if parsed:
                            last_progress = parsed

                # Check for preview updates
                if enable_preview and preview_mp4_path:
                    if os.path.exists(preview_mp4_path):
                        current_mtime = os.path.getmtime(preview_mp4_path)
                        if current_mtime > last_preview_mtime:
                            current_preview_list = [preview_mp4_path]
                            last_preview_mtime = current_mtime

                yield all_generated_videos.copy(), current_preview_list, status_text, last_progress

                if process.poll() is not None:
                    # Read remaining output
                    for line in process.stdout:
                        line = line.strip()
                        if line:
                            print(line)
                            parsed = parse_ltx_progress_line(line)
                            if parsed:
                                last_progress = parsed
                    break

            current_process = None
            return_code = process.returncode
            elapsed = time.perf_counter() - start_time

            if return_code == 0 and os.path.exists(output_filename):
                # Metadata is now saved by the backend (ltx_generate_video.py)
                # embedded directly in the video file's comment metadata

                label = f"Seed: {current_seed} | {elapsed:.1f}s"
                all_generated_videos.append((output_filename, label))

                status_text = f"Completed {i+1}/{batch_size}"
                yield all_generated_videos.copy(), current_preview_list, status_text, f"Video saved: {os.path.basename(output_filename)}"
            else:
                error_msg = f"Generation failed (return code: {return_code})"
                yield all_generated_videos.copy(), current_preview_list, error_msg, "Check logs for details"

        except Exception as e:
            current_process = None
            yield all_generated_videos, [], f"Error: {str(e)}", ""
            return

    final_status = f"Completed {batch_size} video(s)" if batch_size > 1 else "Generation complete!"
    yield all_generated_videos, [], final_status, "Done!"


def stop_generation():
    """Signal to stop the current generation."""
    global stop_event
    stop_event.set()
    return "Stopping generation..."


# =============================================================================
# Video Extension (Wan2GP-style)
# =============================================================================

def generate_extension_video(
    # Extension-specific parameters
    ext_input_video: str,
    ext_prompt: str,
    ext_negative_prompt: str,
    ext_extend_seconds: float,
    ext_preserve_seconds: float,
    ext_seed: int,
    ext_steps: int,
    ext_cfg: float,
    ext_preserve_strength: float,
    ext_skip_stage2: bool,
    # Model paths
    ext_checkpoint_path: str,
    ext_distilled_checkpoint: bool,
    ext_gemma_root: str,
    ext_spatial_upsampler_path: str,
    ext_vae_path: str,
    ext_distilled_lora_path: str,
    ext_distilled_lora_strength: float,
    # Memory optimization
    ext_offload: bool,
    ext_enable_fp8: bool,
    ext_enable_dit_block_swap: bool,
    ext_dit_blocks_in_memory: int,
    ext_enable_text_encoder_block_swap: bool,
    ext_text_encoder_blocks_in_memory: int,
    ext_enable_refiner_block_swap: bool,
    ext_refiner_blocks_in_memory: int,
    ext_enable_activation_offload: bool,
    # LoRA
    ext_lora_folder: str,
    ext_user_lora_1: str,
    ext_user_lora_strength_1: float,
    ext_user_lora_stage_1: str,
    ext_user_lora_2: str,
    ext_user_lora_strength_2: float,
    ext_user_lora_stage_2: str,
    ext_user_lora_3: str,
    ext_user_lora_strength_3: float,
    ext_user_lora_stage_3: str,
    ext_user_lora_4: str,
    ext_user_lora_strength_4: float,
    ext_user_lora_stage_4: str,
    # Output
    ext_save_path: str,
    # Batching
    ext_batch_size: int = 1,
) -> Generator[Tuple[list, str, str], None, None]:
    """Generate video extension using Wan2GP-style conditioning approach."""
    global current_process, current_output_filename, stop_event

    stop_event.clear()

    # Validate input video
    if not ext_input_video:
        yield [], "Error: No input video provided", ""
        return

    if not os.path.exists(ext_input_video):
        yield [], f"Error: Input video not found: {ext_input_video}", ""
        return

    # Validate prompt
    if not ext_prompt or not ext_prompt.strip():
        yield [], "Error: Prompt is required", ""
        return

    # Generate random base seed if -1
    base_seed = ext_seed
    if base_seed == -1:
        base_seed = random.randint(0, 2147483647)

    # Create output directory
    os.makedirs(ext_save_path, exist_ok=True)

    batch_count = max(1, int(ext_batch_size))
    all_outputs = []

    for batch_idx in range(batch_count):
        if stop_event.is_set():
            yield all_outputs, "Generation stopped by user", ""
            return

        # Calculate seed for this batch
        current_seed = base_seed + batch_idx

        # Generate output filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_prompt = re.sub(r'[^\w\s-]', '', ext_prompt[:30]).strip().replace(' ', '_')
        output_filename = f"ext_{timestamp}_{safe_prompt}_{current_seed}.mp4"
        output_path = os.path.join(ext_save_path, output_filename)
        current_output_filename = output_path

        batch_status = f"Batch {batch_idx + 1}/{batch_count}" if batch_count > 1 else ""

        # Build command for ltx_video_extend.py
        command = [
            sys.executable, "ltx_video_extend.py",
            "--input", ext_input_video,
            "--output", output_path,
            "--prompt", ext_prompt,
            "--extend-seconds", str(ext_extend_seconds),
            "--seed", str(current_seed),
            "--steps", str(ext_steps),
            "--cfg", str(ext_cfg),
            "--preserve-strength", str(ext_preserve_strength),
        ]

        # Add preserve-seconds if specified (limit input video context)
        if ext_preserve_seconds and float(ext_preserve_seconds) > 0:
            command.extend(["--preserve-seconds", str(float(ext_preserve_seconds))])

        # Add negative prompt if provided
        if ext_negative_prompt and ext_negative_prompt.strip():
            command.extend(["--negative-prompt", ext_negative_prompt])

        # Model paths
        if ext_checkpoint_path:
            command.extend(["--checkpoint", ext_checkpoint_path])
        if ext_gemma_root:
            command.extend(["--gemma-root", ext_gemma_root])
        if ext_spatial_upsampler_path:
            command.extend(["--spatial-upsampler", ext_spatial_upsampler_path])

        # Distilled LoRA (only if not using distilled checkpoint)
        if not ext_distilled_checkpoint and ext_distilled_lora_path and ext_distilled_lora_path.strip() and os.path.exists(ext_distilled_lora_path):
            command.extend(["--distilled-lora", ext_distilled_lora_path])

        # Skip stage 2
        if ext_skip_stage2:
            command.append("--skip-stage2")

        # Memory optimization
        if ext_enable_dit_block_swap:
            command.append("--dit-block-swap")
            command.extend(["--dit-blocks", str(ext_dit_blocks_in_memory)])

        if ext_enable_text_encoder_block_swap:
            command.append("--text-encoder-block-swap")
            command.extend(["--text-encoder-blocks", str(ext_text_encoder_blocks_in_memory)])

        if ext_enable_refiner_block_swap:
            command.append("--refiner-block-swap")
            command.extend(["--refiner-blocks", str(ext_refiner_blocks_in_memory)])

        if ext_enable_activation_offload:
            command.append("--activation-offload")

        # User LoRAs
        lora_configs = [
            (ext_user_lora_1, ext_user_lora_strength_1, ext_user_lora_stage_1),
            (ext_user_lora_2, ext_user_lora_strength_2, ext_user_lora_stage_2),
            (ext_user_lora_3, ext_user_lora_strength_3, ext_user_lora_stage_3),
            (ext_user_lora_4, ext_user_lora_strength_4, ext_user_lora_stage_4),
        ]
        for user_lora, user_lora_strength, user_lora_stage in lora_configs:
            if user_lora and user_lora != "None" and ext_lora_folder:
                lora_path = os.path.join(ext_lora_folder, user_lora)
                if os.path.exists(lora_path):
                    if user_lora_stage == "Stage 1 (Base)":
                        command.extend(["--lora", lora_path, str(user_lora_strength)])
                    elif user_lora_stage == "Stage 2 (Refine)":
                        command.extend(["--stage2-lora", lora_path, str(user_lora_strength)])
                    else:  # Both
                        command.extend(["--lora", lora_path, str(user_lora_strength)])
                        command.extend(["--stage2-lora", lora_path, str(user_lora_strength)])

        # Print command for debugging
        print("\n" + "=" * 80)
        print(f"LAUNCHING EXTENSION COMMAND (Batch {batch_idx + 1}/{batch_count}):")
        print(" ".join(command))
        print("=" * 80 + "\n")

        yield all_outputs, f"{batch_status} Starting video extension...", f"Command: {' '.join(command[:10])}..."

        try:
            # Use PYTHONUNBUFFERED to ensure subprocess output isn't block-buffered
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            # Start the subprocess
            current_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env
            )

            last_progress = "Starting..."

            # Stream output
            while True:
                if stop_event.is_set():
                    current_process.terminate()
                    try:
                        current_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        current_process.kill()
                    yield all_outputs, "Generation stopped by user", last_progress
                    return

                line = current_process.stdout.readline()
                if line:
                    line = line.strip()
                    if line:
                        print(line)  # Print to terminal
                        # Parse progress
                        progress = parse_ltx_progress_line(line)
                        if progress:
                            last_progress = progress

                yield all_outputs, f"{batch_status} Extending video: {last_progress}", last_progress

                if current_process.poll() is not None:
                    # Read remaining output
                    for line in current_process.stdout:
                        line = line.strip()
                        if line:
                            print(line)  # Print to terminal
                            progress = parse_ltx_progress_line(line)
                            if progress:
                                last_progress = progress
                    break

            return_code = current_process.returncode
            if return_code == 0 and os.path.exists(output_path):
                all_outputs.append(output_path)
                if batch_idx == batch_count - 1:
                    yield all_outputs, f"All {batch_count} extension(s) complete!", "Done!"
                else:
                    yield all_outputs, f"{batch_status} Complete. Starting next...", "Done"
            else:
                yield all_outputs, f"{batch_status} Extension failed (code {return_code}). Check terminal for details.", "Error"

        except Exception as e:
            yield all_outputs, f"{batch_status} Extension error: {str(e)}", "Error"
        finally:
            current_process = None
            current_output_filename = None


# =============================================================================
# SVI-LTX Frame Checking Functions
# =============================================================================

def variance_of_laplacian(image):
    """Calculate image sharpness using Laplacian variance."""
    import cv2
    return cv2.Laplacian(image, cv2.CV_64F).var()


def extract_best_transition_frame(video_path: str, frames_to_check: int = 30) -> int:
    """
    Find the sharpest frame from the last N frames for smooth transition.
    Uses Laplacian variance as sharpness metric.

    Args:
        video_path: Path to video file
        frames_to_check: Number of frames from end to analyze

    Returns:
        Index of the best (sharpest) frame
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return -1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, total_frames - frames_to_check)

    best_frame_idx = total_frames - 1
    best_sharpness = -1

    for frame_idx in range(start_frame, total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = variance_of_laplacian(gray)
        if sharpness > best_sharpness:
            best_sharpness = sharpness
            best_frame_idx = frame_idx

    cap.release()
    return best_frame_idx


# =============================================================================
# SVI-LTX Video Generation Handler
# =============================================================================

def generate_svi_ltx_video(
    # Prompts (8 clips)
    prompt1: str, prompt2: str, prompt3: str, prompt4: str,
    prompt5: str, prompt6: str, prompt7: str, prompt8: str,
    negative_prompt: str,
    # Images
    input_image: str,
    anchor_image: str,
    # Video extension
    extend_video: str,
    frames_to_check: int,
    prepend_original: bool,
    # SVI settings
    num_clips: int,
    overlap_frames: int,
    num_motion_latent: int,
    num_motion_frame: int,
    seed_multiplier: int,
    # Generation params
    width: int,
    height: int,
    num_frames: int,
    frame_rate: float,
    inference_steps: int,
    cfg_scale: float,
    seed: int,
    # Anchor settings
    anchor_interval: int,
    anchor_strength: float,
    anchor_decay: str,
    # Model settings
    checkpoint_path: str,
    gemma_root: str,
    spatial_upsampler: str,
    distilled_lora: str,
    distilled_lora_strength: float,
    one_stage: bool,
    enable_fp8: bool,
    offload: bool,
    enable_dit_block_swap: bool,
    dit_blocks_in_memory: int,
    enable_text_encoder_block_swap: bool,
    text_encoder_blocks_in_memory: int,
    enable_refiner_block_swap: bool,
    refiner_blocks_in_memory: int,
    ffn_chunk_size: int,
    enable_activation_offload: bool,
    temporal_chunk_size: int,
    # LoRA
    lora_folder: str,
    user_lora: str,
    lora_strength: float,
    # Output
    disable_audio: bool,
    output_path: str,
    batch_size: int,
    # Preview Generation
    enable_preview: bool,
    preview_interval: int,
) -> Generator[Tuple[List[Tuple[str, str]], Optional[str], str, str], None, None]:
    """
    Generate multi-clip SVI video using LTX-2 pipeline.

    Yields: (gallery_items, preview_path, status_text, progress_text)
    """
    global stop_event, current_process, current_output_filename
    stop_event.clear()
    current_process = None
    current_output_filename = None

    # Validate inputs
    if not extend_video and not input_image:
        yield [], None, "Error: Input image required for SVI multi-clip mode", ""
        return

    error = validate_num_frames(int(num_frames))
    if error:
        yield [], None, error, ""
        return

    error = validate_resolution(int(width), int(height))
    if error:
        yield [], None, error, ""
        return

    # Round num_frames to valid 8*K+1 format
    num_frames = int(num_frames)
    k = max(1, round((num_frames - 1) / 8))
    num_frames = 8 * k + 1

    os.makedirs(output_path, exist_ok=True)
    all_generated_videos = []

    for batch_idx in range(int(batch_size)):
        if stop_event.is_set():
            current_process = None
            yield all_generated_videos, None, "Generation stopped by user.", ""
            return

        # Seed handling
        current_seed = seed
        if seed == -1:
            current_seed = random.randint(0, 2**32 - 1)
        elif int(batch_size) > 1:
            current_seed = seed + batch_idx * 1000

        # Collect per-clip prompts (non-empty only)
        all_prompts = [prompt1, prompt2, prompt3, prompt4, prompt5, prompt6, prompt7, prompt8]
        clip_prompts = [p.strip() for p in all_prompts if p and p.strip()]
        main_prompt = clip_prompts[0] if clip_prompts else "A beautiful video"

        status_text = f"Processing SVI {batch_idx+1}/{batch_size} ({num_clips} clips, Seed: {current_seed})"
        yield all_generated_videos.copy(), None, status_text, "Starting SVI generation..."

        timestamp = int(time.time())
        run_id = f"{timestamp}_{random.randint(1000, 9999)}"
        mode_str = "svi_extend" if extend_video else "svi"
        output_filename = os.path.join(output_path, f"ltx_{mode_str}_{timestamp}_{current_seed}.mp4")
        current_output_filename = output_filename

        # Frame checking for video extension
        if extend_video:
            best_frame = extract_best_transition_frame(extend_video, int(frames_to_check))
            print(f">>> Best transition frame: {best_frame}")

        # Build command
        command = [
            sys.executable, "ltx_generate_video.py",
            "--checkpoint-path", checkpoint_path,
            "--gemma-root", gemma_root,
            "--prompt", str(main_prompt),
            "--negative-prompt", str(negative_prompt),
            "--num-frames", str(int(num_frames)),
            "--frame-rate", str(float(frame_rate)),
            "--width", str(int(width)),
            "--height", str(int(height)),
            "--cfg-guidance-scale", str(float(cfg_scale)),
            "--num-inference-steps", str(int(inference_steps)),
            "--seed", str(current_seed),
            "--output-path", output_filename,
        ]

        # SVI mode or video extension
        if extend_video:
            command.extend(["--extend-video", str(extend_video)])
            if prepend_original:
                command.append("--prepend-original")
        else:
            command.append("--svi-mode")
            if input_image:
                command.extend(["--image", str(input_image), "0", "0.95"])

        # SVI settings
        command.extend([
            "--num-clips", str(int(num_clips)),
            "--num-motion-latent", str(int(num_motion_latent)),
            "--num-motion-frame", str(int(num_motion_frame)),
            "--seed-multiplier", str(int(seed_multiplier)),
            "--overlap-frames", str(int(overlap_frames)),
        ])

        # Per-clip prompts (if more than one provided)
        if len(clip_prompts) > 1:
            command.append("--prompt-list")
            command.extend(clip_prompts)

        # Anchor settings
        if anchor_image:
            command.extend(["--anchor-image", str(anchor_image)])
        if anchor_interval and int(anchor_interval) > 0:
            command.extend([
                "--anchor-interval", str(int(anchor_interval)),
                "--anchor-strength", str(float(anchor_strength)),
            ])
            if anchor_decay and anchor_decay != "none":
                command.extend(["--anchor-decay", str(anchor_decay)])

        # Model settings
        if not one_stage:
            if spatial_upsampler and os.path.exists(spatial_upsampler):
                command.extend(["--spatial-upsampler-path", spatial_upsampler])
            if distilled_lora and os.path.exists(distilled_lora):
                command.extend(["--distilled-lora", distilled_lora, str(distilled_lora_strength)])
        else:
            command.append("--one-stage")

        if enable_fp8:
            command.append("--enable-fp8")
        if offload:
            command.append("--offload")

        # Block swap settings
        if enable_dit_block_swap:
            command.append("--enable-dit-block-swap")
            command.extend(["--dit-blocks-in-memory", str(int(dit_blocks_in_memory))])
        if enable_text_encoder_block_swap:
            command.append("--enable-text-encoder-block-swap")
            command.extend(["--text-encoder-blocks-in-memory", str(int(text_encoder_blocks_in_memory))])
        if enable_refiner_block_swap:
            command.append("--enable-refiner-block-swap")
            command.extend(["--refiner-blocks-in-memory", str(int(refiner_blocks_in_memory))])
        if ffn_chunk_size and int(ffn_chunk_size) > 0:
            command.extend(["--ffn-chunk-size", str(int(ffn_chunk_size))])

        # User LoRA
        if user_lora and user_lora != "None" and lora_folder:
            lora_path = os.path.join(lora_folder, user_lora)
            if os.path.exists(lora_path):
                command.extend(["--lora", lora_path, str(lora_strength)])

        # Audio
        if disable_audio:
            command.append("--disable-audio")

        # Preview generation
        unique_preview_suffix = f"svi_{run_id}"
        preview_mp4_path = None
        if enable_preview and preview_interval > 0:
            preview_base_dir = os.path.join(output_path, "previews")
            command.extend(["--preview-dir", preview_base_dir])
            command.extend(["--preview-interval", str(int(preview_interval))])
            command.extend(["--preview-suffix", unique_preview_suffix])
            preview_mp4_path = os.path.join(preview_base_dir, f"latent_preview_{unique_preview_suffix}.mp4")

        # Print command for debugging
        print("\n" + "=" * 80)
        print(f"LAUNCHING SVI COMMAND (Batch {batch_idx+1}/{batch_size}):")
        print(" ".join(command))
        print("=" * 80 + "\n")

        try:
            start_time = time.perf_counter()

            # Get script directory to ensure ltx_generate_video.py is found
            script_dir = os.path.dirname(os.path.abspath(__file__))
            print(f">>> Running from: {script_dir}")

            # Use PYTHONUNBUFFERED to ensure subprocess output isn't block-buffered
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=script_dir,  # Run from the script directory
                env=env
            )

            current_process = process
            last_progress = ""
            current_preview_list = []
            last_preview_mtime = 0

            while True:
                if stop_event.is_set():
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    current_process = None
                    yield all_generated_videos, [], "Generation stopped by user.", ""
                    return

                # Read output line by line (blocking but with PYTHONUNBUFFERED)
                line = process.stdout.readline()
                if line:
                    line = line.strip()
                    if line:
                        print(line)
                        progress_info = parse_ltx_progress_line(line)
                        if progress_info:
                            last_progress = progress_info

                # Check for preview updates
                if enable_preview and preview_mp4_path:
                    if os.path.exists(preview_mp4_path):
                        current_mtime = os.path.getmtime(preview_mp4_path)
                        if current_mtime > last_preview_mtime:
                            current_preview_list = [preview_mp4_path]
                            last_preview_mtime = current_mtime

                exit_code = process.poll()

                if exit_code is not None:
                    # Read remaining output
                    for line in process.stdout:
                        line = line.strip()
                        if line:
                            print(line)
                            progress_info = parse_ltx_progress_line(line)
                            if progress_info:
                                last_progress = progress_info

                    if exit_code == 0 and os.path.exists(output_filename):
                        elapsed = time.perf_counter() - start_time
                        label = f"SVI {num_clips} clips ({elapsed:.1f}s)"
                        all_generated_videos.append((output_filename, label))
                        yield all_generated_videos.copy(), current_preview_list, f"Completed batch {batch_idx+1}/{batch_size}", "Done!"
                    else:
                        error_msg = f"Error in batch {batch_idx+1} (exit code {exit_code})"
                        yield all_generated_videos, current_preview_list, error_msg, ""
                    break

                yield all_generated_videos.copy(), current_preview_list, status_text, last_progress

        except Exception as e:
            yield all_generated_videos, [], f"Error: {str(e)}", ""
            continue

    final_status = f"Completed {batch_size} SVI video(s)" if batch_size > 1 else "SVI generation complete!"
    yield all_generated_videos, [], final_status, "Done!"


# =============================================================================
# Depth Map Generation
# =============================================================================

def generate_depth_map(
    input_type: str,
    input_image: str,
    input_video: str,
    output_type: str,
    colorize: bool,
    num_frames: int,
    fps: float,
    max_frames: int,
    width: int,
    height: int,
) -> Generator[Tuple[Optional[str], Optional[str], str], None, None]:
    """
    Generate depth map from image or video using ZoeDepth.

    Yields:
        (image_output, video_output, status)
    """
    import tempfile

    # Validate input
    input_path = input_image if input_type == "Image" else input_video
    if not input_path:
        yield None, None, "Error: No input file provided"
        return

    if not os.path.exists(input_path):
        yield None, None, f"Error: Input file not found: {input_path}"
        return

    yield None, None, "Loading depth model..."

    # Determine output path
    output_dir = tempfile.gettempdir()
    timestamp = int(time.time())

    if output_type == "Video" or (input_type == "Video" and output_type != "Image"):
        output_path = os.path.join(output_dir, f"depth_{timestamp}.mp4")
    else:
        output_path = os.path.join(output_dir, f"depth_{timestamp}.png")

    # Build command
    script_dir = os.path.dirname(os.path.abspath(__file__))
    command = [
        sys.executable,
        os.path.join(script_dir, "depth_map_generator.py"),
        "--input", input_path,
        "--output", output_path,
    ]

    if colorize:
        command.append("--colorize")

    if input_type == "Image" and output_type == "Video":
        command.extend(["--num-frames", str(int(num_frames))])
        command.extend(["--fps", str(fps)])

    if input_type == "Video" and max_frames > 0:
        command.extend(["--max-frames", str(int(max_frames))])

    if width > 0 and height > 0:
        command.extend(["--width", str(int(width))])
        command.extend(["--height", str(int(height))])

    yield None, None, "Generating depth map..."

    try:
        # Run the depth generation script
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=script_dir,
            env=env
        )

        # Read output for progress
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if line:
                # Print to console
                print(line, flush=True)
                # Parse progress for GUI
                if "Loading depth model" in line:
                    yield None, None, "Loading ZoeDepth model..."
                elif "Estimating depth" in line:
                    yield None, None, line
                elif "Saving" in line:
                    yield None, None, "Saving output..."
                elif "%" in line:
                    yield None, None, line

        process.wait()

        if process.returncode != 0:
            yield None, None, f"Error: Depth generation failed (exit code {process.returncode})"
            return

        # Return appropriate output
        if output_path.endswith(".mp4"):
            yield None, output_path, f"Depth video saved: {output_path}"
        else:
            yield output_path, None, f"Depth image saved: {output_path}"

    except Exception as e:
        yield None, None, f"Error: {str(e)}"


# =============================================================================
# Gradio Interface
# =============================================================================

def create_interface():
    with gr.Blocks(
        theme=themes.Default(
            primary_hue=colors.Color(
                name="ltx_blue",
                c50="#E6F4FF",
                c100="#BAE0FF",
                c200="#91CAFF",
                c300="#69B4FF",
                c400="#409EFF",
                c500="#1677FF",
                c600="#0958D9",
                c700="#003EB3",
                c800="#002C8C",
                c900="#001D66",
                c950="#000F33"
            )
        ),
        css="""
        .green-btn {
            background: linear-gradient(to bottom right, #2ecc71, #27ae60) !important;
            color: white !important;
            border: none !important;
        }
        .green-btn:hover {
            background: linear-gradient(to bottom right, #27ae60, #219651) !important;
        }
        """,
        title="LTX-2 Video Generator"
    ) as demo:
        with gr.Tabs() as tabs:
            # =================================================================
            # Generation Tab
            # =================================================================
            with gr.Tab("Generation", id="gen_tab"):
                with gr.Row():
                    # Left column - Inputs
                    with gr.Column(scale=3):
                        prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe the video you want to generate...",
                            lines=4,
                            value="A giant brown capybara wearing white wireless earbuds stands upright on a Miami beach, mouth wide open revealing large front teeth, swaying its round body rhythmically side to side. Beside it, a gray tabby cat with green eyes wears a pink knit beanie with pom-pom, black bow tie, light blue t-shirt with small animal graphic, and blue jeans, pumping two bright green glow sticks overhead in time with pulsing electronic beats. The duo bounces and grooves together on golden sand scattered with seashells, their movements becoming more energetic as synthesizer drops hit. Palm trees sway in the warm breeze behind them, with Miami high-rise buildings visible in the distance. The sky transitions dramatically from warm orange and pink sunset hues to deep purple twilight, as a bright full moon rises over the turquoise ocean waves. Colorful beach umbrellas in red and white dot the background. The camera circles slowly around the dancing pair in a smooth 360-degree arc, capturing their joyful expressions. Lens flares catch the fading sunlight, then neon glow stick trails create light streaks as darkness falls. The scene pulses with energy as city lights begin twinkling along the coastline, waves crashing rhythmically in the moonlit background."
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            value="blurry, low quality, distorted, artifacts, ugly, deformed",
                            lines=2
                        )

                        with gr.Row():
                            batch_size = gr.Number(label="Batch Count", value=1, minimum=1, step=1, scale=1)
                            seed = gr.Number(label="Seed (-1 = random)", value=-1, scale=1)
                            random_seed_btn = gr.Button("🎲", scale=0, min_width=40)

                    # Right column - Status
                    with gr.Column(scale=1):
                        status_text = gr.Textbox(label="Status", interactive=False, value="Ready")
                        progress_text = gr.Textbox(label="Progress", interactive=False, value="")

                with gr.Row():
                    generate_btn = gr.Button("🎬 Generate Video", variant="primary", elem_classes="green-btn")
                    stop_btn = gr.Button("⏹️ Stop", variant="stop")

                with gr.Row():
                    # Left column - Parameters
                    with gr.Column():
                        # Hidden state for original image/video dimensions
                        original_dims = gr.State(value="")

                        # Image Conditioning (I2V) - moved above Resolution Settings
                        with gr.Accordion("Image Conditioning (I2V)", open=True) as i2v_section:
                            input_image = gr.Image(label="Start Image", type="filepath")
                            with gr.Row():
                                image_frame_idx = gr.Number(label="Frame Index", value=0, minimum=0, info="Which frame to condition (0 = first)")
                                image_strength = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.05, label="Strength")
                                image_crf = gr.Slider(minimum=0, maximum=51, value=33, step=1, label="CRF", info="H.264 compression (0=lossless, 33=default)")

                            with gr.Accordion("Anchor Image (periodic guidance)", open=False):
                                gr.Markdown("Inject the anchor image at regular intervals to guide the video generation. Small strength values work best, like .01-.05")
                                anchor_image = gr.Image(label="Anchor Image (optional, uses Start Image if empty)", type="filepath")
                                with gr.Row():
                                    anchor_interval = gr.Number(
                                        label="Anchor Interval",
                                        value=0,
                                        minimum=0,
                                        step=8,
                                        info="Frame interval (e.g., 60). Set to 0 to disable."
                                    )
                                    anchor_strength = gr.Slider(
                                        minimum=0.0, maximum=1.0, value=0.1, step=0.001,
                                        label="Anchor Strength",
                                        info="How strongly to guide toward anchor"
                                    )
                                    anchor_decay = gr.Dropdown(
                                        label="Anchor Decay",
                                        choices=["none", "linear", "cosine", "sigmoid"],
                                        value="cosine",
                                        info="Decay schedule: strong early, weak later for motion"
                                    )
                                    anchor_crf = gr.Slider(minimum=0, maximum=51, value=33, step=1, label="CRF", info="H.264 compression (0=lossless, 33=default)")

                            with gr.Accordion("End Image (optional)", open=False):
                                end_image = gr.Image(label="End Image (for start-to-end video)", type="filepath")
                                gr.Markdown("Set an ending frame to generate video that transitions from start to end image.")
                                with gr.Row():
                                    end_image_strength = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.05, label="End Image Strength")
                                    end_image_crf = gr.Slider(minimum=0, maximum=51, value=33, step=1, label="CRF", info="H.264 compression (0=lossless, 33=default)")

                        # Resolution Settings (always visible, outside accordions)
                        gr.Markdown("### Resolution Settings")
                        with gr.Row():
                            mode = gr.Dropdown(
                                label="Mode",
                                choices=["t2v", "i2v", "v2v"],
                                value="t2v",
                                info="t2v = text-to-video, i2v = image-to-video, v2v = video-to-video (refine)"
                            )
                            pipeline = gr.Dropdown(
                                label="Pipeline",
                                choices=["two-stage", "one-stage", "refine-only", "three-stage-exp", "three-stage-linear"],
                                value="two-stage",
                                info="two-stage = higher quality, three-stage-exp = 3 stages with upsampling, three-stage-linear = 3 stages same res"
                            )
                            sampler = gr.Dropdown(
                                label="Sampler (Stage 1)",
                                choices=["euler", "unipc", "lcm"],
                                value="euler",
                                info="euler = default, unipc = higher-order (10-25 steps), lcm = fast (4-8 steps)"
                            )
                            stage2_sampler = gr.Dropdown(
                                label="Sampler (Stage 2/3)",
                                choices=["euler", "unipc", "lcm"],
                                value="euler",
                                info="euler recommended for distilled models with few steps"
                            )
                        with gr.Row():
                            stage2_strength = gr.Slider(
                                minimum=0.01, maximum=1.0, value=1.0, step=0.01,
                                label="Stage 2 Strength",
                                info="Sigma scaling (lower = less noise = more preservation)"
                            )
                            stage3_strength = gr.Slider(
                                minimum=0.01, maximum=1.0, value=1.0, step=0.01,
                                label="Stage 3 Strength",
                                info="Sigma scaling (lower = less noise = more preservation)"
                            )
                        scale_slider = gr.Slider(
                            minimum=1, maximum=200, value=100, step=1,
                            label="Scale % (adjusts resolution while maintaining aspect ratio)",
                            info="Scale the input image dimensions. Works for I2V mode."
                        )
                        with gr.Row():
                            width = gr.Number(label="Width", value=1024, step=64, info="Must be divisible by 64")
                            calc_height_btn = gr.Button("→", size="sm", min_width=40)
                            calc_width_btn = gr.Button("←", size="sm", min_width=40)
                            height = gr.Number(label="Height", value=1024, step=64, info="Must be divisible by 64")
                        with gr.Row():
                            num_frames = gr.Slider(minimum=9, maximum=2001, step=8, value=121, label="Num Frames (8*K+1)", info="e.g., 121 = 5s @ 24fps")
                            frame_rate = gr.Slider(minimum=12, maximum=60, value=24, step=1, label="Frame Rate")
                        with gr.Row():
                            cfg_guidance_scale = gr.Slider(minimum=1.0, maximum=15.0, value=4.0, step=0.5, label="CFG Scale")
                            num_inference_steps = gr.Slider(minimum=1, maximum=60, value=40, step=1, label="Inference Steps")
                            stage2_steps = gr.Slider(minimum=1, maximum=60, value=3, step=1, label="Stage 2 Steps")
                        with gr.Row():
                            stg_scale = gr.Slider(minimum=0.0, maximum=2.0, value=0.0, step=0.1, label="STG Scale", info="Spatio-temporal guidance scale (0=disabled)")
                            stg_blocks = gr.Textbox(label="STG Blocks", value="29", info="Comma-separated block indices, e.g., 29 or 20,21,22")
                            stg_mode = gr.Dropdown(label="STG Mode", choices=["stg_av", "stg_v"], value="stg_av", info="stg_av=audio+video, stg_v=video only")

                        # Advanced CFG (MultiModal Guidance)
                        with gr.Accordion("Advanced CFG (MultiModal Guidance)", open=False):
                            gr.Markdown("""
                            **MultiModal Guidance Mode:**
                            Enables separate video/audio guidance controls with advanced features:
                            - **Rescale Scale**: Variance normalization to prevent oversaturation
                            - **Modality Scale**: Cross-modal guidance (A2V for video, V2A for audio)
                            - **Skip Step**: Periodic step skipping for speedup
                            """)
                            guidance_mode = gr.Dropdown(
                                label="Guidance Mode",
                                choices=["legacy", "multimodal"],
                                value="legacy",
                                info="'legacy'=standard CFG+STG, 'multimodal'=separate video/audio control"
                            )
                            scheduler_type = gr.Dropdown(
                                label="Scheduler",
                                choices=["ltx2", "linear_quadratic", "beta"],
                                value="ltx2",
                                info="Sigma schedule type for diffusion sampling"
                            )
                            with gr.Row():
                                kandinsky_scheduler = gr.Checkbox(label="Kandinsky Scheduler", value=False, info="Front-loaded scheduler for better fast motion (recommended: 50+ steps)")
                                kandinsky_scheduler_scale = gr.Slider(minimum=1.0, maximum=10.0, value=5.0, step=0.5, label="Kandinsky Scale", info="Higher = more motion focus (3-7 recommended)")
                            with gr.Row():
                                gr.Markdown("### Video Guidance")
                            with gr.Row():
                                video_cfg_guidance_scale = gr.Slider(
                                    minimum=1.0, maximum=15.0, value=3.0, step=0.5,
                                    label="Video CFG Scale",
                                    info="Higher = more prompt adherence"
                                )
                                video_stg_scale = gr.Slider(
                                    minimum=0.0, maximum=2.0, value=1.0, step=0.1,
                                    label="Video STG Scale",
                                    info="Spatio-temporal guidance for video"
                                )
                            with gr.Row():
                                video_stg_blocks = gr.Textbox(
                                    label="Video STG Blocks",
                                    value="29",
                                    info="Comma-separated block indices"
                                )
                                video_rescale_scale = gr.Slider(
                                    minimum=0.0, maximum=1.0, value=0.7, step=0.05,
                                    label="Video Rescale",
                                    info="Variance normalization (0=off)"
                                )
                            with gr.Row():
                                a2v_guidance_scale = gr.Slider(
                                    minimum=1.0, maximum=10.0, value=3.0, step=0.5,
                                    label="A2V Guidance Scale",
                                    info="Audio-to-Video cross-modal guidance"
                                )
                                video_skip_step = gr.Slider(
                                    minimum=0, maximum=5, value=0, step=1,
                                    label="Video Skip Step",
                                    info="Skip every N+1 steps (0=none)"
                                )
                            with gr.Row():
                                gr.Markdown("### Audio Guidance")
                            with gr.Row():
                                audio_cfg_guidance_scale = gr.Slider(
                                    minimum=1.0, maximum=15.0, value=7.0, step=0.5,
                                    label="Audio CFG Scale",
                                    info="Higher = more prompt adherence"
                                )
                                audio_stg_scale = gr.Slider(
                                    minimum=0.0, maximum=2.0, value=1.0, step=0.1,
                                    label="Audio STG Scale",
                                    info="Spatio-temporal guidance for audio"
                                )
                            with gr.Row():
                                audio_stg_blocks = gr.Textbox(
                                    label="Audio STG Blocks",
                                    value="29",
                                    info="Comma-separated block indices"
                                )
                                audio_rescale_scale = gr.Slider(
                                    minimum=0.0, maximum=1.0, value=0.7, step=0.05,
                                    label="Audio Rescale",
                                    info="Variance normalization (0=off)"
                                )
                            with gr.Row():
                                v2a_guidance_scale = gr.Slider(
                                    minimum=1.0, maximum=10.0, value=3.0, step=0.5,
                                    label="V2A Guidance Scale",
                                    info="Video-to-Audio cross-modal guidance"
                                )
                                audio_skip_step = gr.Slider(
                                    minimum=0, maximum=5, value=0, step=1,
                                    label="Audio Skip Step",
                                    info="Skip every N+1 steps (0=none)"
                                )

                        # Video Input (V2V / Refine)
                        with gr.Accordion("Video Input (V2V / Refine)", open=False) as v2v_section:
                            input_video = gr.Video(label="Input Video (for refinement)", sources=["upload"])
                            gr.Markdown("""
                            **Video-to-Video Refinement:**
                            - Upload a video to refine it using stage 2 (with distilled LoRA)
                            - Use "refine-only" pipeline to skip stage 1 generation
                            - Resolution and frame count will be auto-detected from video
                            """)
                            with gr.Row():
                                refine_strength = gr.Slider(
                                    minimum=0.0, maximum=1.0, value=0.3, step=0.05,
                                    label="Refine Strength",
                                    info="Amount of noise to add before refinement (0=none, 1=full denoise)"
                                )

                        # Depth Control (IC-LoRA)
                        with gr.Accordion("Depth Control (IC-LoRA)", open=False):
                            gr.Markdown("""
                            **Depth-guided video generation using IC-LoRA.**
                            Load the depth control LoRA and provide depth maps to guide video structure.
                            Generate depth maps in the **Depth Map** tab.
                            """)
                            with gr.Row():
                                depth_control_video = gr.Video(
                                    label="Depth Map Video (pre-generated)",
                                    sources=["upload"]
                                )
                                depth_control_image = gr.Image(
                                    label="Depth Map Image (applied to all frames)",
                                    type="filepath"
                                )
                            with gr.Row():
                                estimate_depth = gr.Checkbox(
                                    label="Auto-estimate Depth",
                                    value=False,
                                    info="Estimate depth from input image/video using ZoeDepth"
                                )
                                depth_strength = gr.Slider(
                                    minimum=0.0, maximum=1.0, value=1.0, step=0.05,
                                    label="Depth Strength",
                                    info="Conditioning strength (1.0 = full)"
                                )
                                depth_stage2 = gr.Checkbox(
                                    label="Apply to Stage 2",
                                    value=False,
                                    info="Also apply depth to refinement stage"
                                )
                            depth_control_status = gr.Textbox(
                                label="Depth Map Info",
                                value="No depth map loaded",
                                interactive=False
                            )

                        # Audio Conditioning
                        with gr.Accordion("Audio Conditioning", open=False):
                            gr.Markdown("Control audio handling for V2V mode and external audio conditioning.")
                            # V2V Audio Mode (only relevant when input_video is provided)
                            v2v_audio_mode = gr.Dropdown(
                                choices=["preserve", "condition", "regenerate", "external"],
                                value="preserve",
                                label="V2V Audio Mode",
                                info="preserve=lock original, condition=use as soft conditioning, regenerate=new audio, external=use file below"
                            )
                            v2v_audio_strength = gr.Slider(
                                minimum=0.0, maximum=1.0, value=1.0, step=0.05,
                                label="V2V Audio Strength",
                                info="1.0 = frozen/exact, 0.0 = regenerate with guidance (only for 'condition' mode)",
                                visible=False
                            )
                            gr.Markdown("---")
                            gr.Markdown("**External Audio File** (used when V2V mode is 'external' or for T2V/I2V)")
                            audio_input = gr.Audio(
                                label="Audio File",
                                type="filepath",
                                sources=["upload"]
                            )
                            audio_strength = gr.Slider(
                                minimum=0.0, maximum=1.0, value=1.0, step=0.05,
                                label="Audio Strength",
                                info="1.0 = frozen/exact audio, 0.0 = regenerate audio"
                            )

                        # Video Continuation (Frame Freezing)
                        with gr.Accordion("Video Continuation (Frame Freezing)", open=False):
                            gr.Markdown("Freeze first N frames from input video during denoising for smooth continuation.")
                            with gr.Row():
                                freeze_frames = gr.Slider(
                                    minimum=0, maximum=32, value=0, step=1,
                                    label="Freeze Frames",
                                    info="Number of frames to freeze from input video (0 = disabled)"
                                )
                                freeze_transition = gr.Slider(
                                    minimum=1, maximum=16, value=4, step=1,
                                    label="Transition Frames",
                                    info="Frames for gradual blend from frozen to generated"
                                )

                        # Sliding Window (Long Video)
                        with gr.Accordion("Sliding Window (Long Video)", open=False):
                            enable_sliding_window = gr.Checkbox(
                                label="Enable Sliding Window",
                                value=False,
                                info="Enable for long videos (>129 frames)"
                            )
                            gr.Markdown("Generate videos longer than the model's context window by overlapping windows.")
                            with gr.Row():
                                sliding_window_size = gr.Slider(
                                    minimum=0, maximum=257, value=0, step=8,
                                    label="Window Size",
                                    info="Frames per window (8n+1). 0 = auto-detect from num_frames"
                                )
                                sliding_window_overlap = gr.Slider(
                                    minimum=1, maximum=33, value=9, step=8,
                                    label="Overlap Frames",
                                    info="Overlapping frames between windows (8n+1)"
                                )
                            with gr.Row():
                                sliding_window_overlap_noise = gr.Slider(
                                    minimum=0.0, maximum=100.0, value=0.0, step=5.0,
                                    label="Overlap Noise %",
                                    info="Noise level for overlap blending (0 = no noise)"
                                )
                                sliding_window_color_correction = gr.Slider(
                                    minimum=0.0, maximum=1.0, value=0.0, step=0.1,
                                    label="Color Correction",
                                    info="LAB color correction strength between windows"
                                )

                        # AV Extension (Time-Based Audio-Video Continuation)
                        with gr.Accordion("AV Extension (Audio-Video Continuation)", open=False):
                            gr.Markdown("""
**Time-based audio-video continuation** - Use t2v mode.
                            """)
                            av_extend_video = gr.Video(label="Input Video to Extend", sources=["upload"])
                            with gr.Row():
                                av_extend_start_time = gr.Number(
                                    label="Start Time (seconds)",
                                    value=0,
                                    minimum=0,
                                    maximum=300,
                                    info="Time to start generating new content. 0 = auto (end of video)"
                                )
                                av_extend_end_time = gr.Number(
                                    label="End Time (seconds)",
                                    value=0,
                                    minimum=0,
                                    maximum=300,
                                    info="Time to stop generation. 0 = auto (start + 5 seconds)"
                                )
                            with gr.Row():
                                av_extend_steps = gr.Slider(
                                    minimum=4, maximum=60, value=8, step=1,
                                    label="Extension Steps",
                                )
                                av_extend_terminal = gr.Slider(
                                    minimum=0.0, maximum=0.5, value=0.1, step=0.01,
                                    label="Terminal Sigma",
                                    info="Smaller = smoother continuation, larger = more creative"
                                )
                            with gr.Row():
                                av_slope_len = gr.Slider(
                                    minimum=1, maximum=16, value=3, step=1,
                                    label="Transition Length",
                                    info="Smoothness at mask boundaries (latent frames)"
                                )
                                av_no_stage2 = gr.Checkbox(
                                    label="Skip Stage 2 Refinement",
                                    value=False,
                                    info="Faster but lower quality"
                                )


                    # Right column - Output
                    with gr.Column():
                        output_gallery = gr.Gallery(
                            label="Generated Videos",
                            columns=2, rows=2,
                            object_fit="contain",
                            height="auto",
                            allow_preview=True,
                            preview=True
                        )
                        # Latent Preview (During Generation)
                        with gr.Accordion("Latent Preview (During Generation)", open=False):
                            enable_preview = gr.Checkbox(label="Enable Latent Preview", value=False)
                            preview_interval = gr.Slider(
                                minimum=1, maximum=50, step=1, value=5,
                                label="Preview Every N Steps"
                            )
                            preview_gallery = gr.Gallery(
                                label="Latent Previews",
                                columns=4, rows=2,
                                object_fit="contain",
                                height=300,
                                allow_preview=True,
                                preview=True,
                                show_label=True
                            )
                        # User LoRAs (up to 4)
                        with gr.Accordion("User LoRAs (Optional)", open=True):
                            lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                            lora_refresh_btn = gr.Button("🔄 Refresh", size="sm")
                            # LoRA 1
                            with gr.Row():
                                user_lora_1 = gr.Dropdown(
                                    label="LoRA 1",
                                    choices=get_ltx_lora_options("lora"),
                                    value="None",
                                    scale=3
                                )
                                user_lora_strength_1 = gr.Slider(
                                    minimum=0.0, maximum=2.0, value=1.0, step=0.1,
                                    label="Strength", scale=1
                                )
                                user_lora_stage_1 = gr.Dropdown(
                                    label="Stage",
                                    choices=["Stage 1 (Base)", "Stage 2 (Refine)", "Stage 3 (Refine)", "All"],
                                    value="Stage 2 (Refine)",
                                    scale=1
                                )
                            # LoRA 2
                            with gr.Row():
                                user_lora_2 = gr.Dropdown(
                                    label="LoRA 2",
                                    choices=get_ltx_lora_options("lora"),
                                    value="None",
                                    scale=3
                                )
                                user_lora_strength_2 = gr.Slider(
                                    minimum=0.0, maximum=2.0, value=1.0, step=0.1,
                                    label="Strength", scale=1
                                )
                                user_lora_stage_2 = gr.Dropdown(
                                    label="Stage",
                                    choices=["Stage 1 (Base)", "Stage 2 (Refine)", "Stage 3 (Refine)", "All"],
                                    value="Stage 2 (Refine)",
                                    scale=1
                                )
                            # LoRA 3
                            with gr.Row():
                                user_lora_3 = gr.Dropdown(
                                    label="LoRA 3",
                                    choices=get_ltx_lora_options("lora"),
                                    value="None",
                                    scale=3
                                )
                                user_lora_strength_3 = gr.Slider(
                                    minimum=0.0, maximum=2.0, value=1.0, step=0.1,
                                    label="Strength", scale=1
                                )
                                user_lora_stage_3 = gr.Dropdown(
                                    label="Stage",
                                    choices=["Stage 1 (Base)", "Stage 2 (Refine)", "Stage 3 (Refine)", "All"],
                                    value="Stage 2 (Refine)",
                                    scale=1
                                )
                            # LoRA 4
                            with gr.Row():
                                user_lora_4 = gr.Dropdown(
                                    label="LoRA 4",
                                    choices=get_ltx_lora_options("lora"),
                                    value="None",
                                    scale=3
                                )
                                user_lora_strength_4 = gr.Slider(
                                    minimum=0.0, maximum=2.0, value=1.0, step=0.1,
                                    label="Strength", scale=1
                                )
                                user_lora_stage_4 = gr.Dropdown(
                                    label="Stage",
                                    choices=["Stage 1 (Base)", "Stage 2 (Refine)", "Stage 3 (Refine)", "All"],
                                    value="Stage 2 (Refine)",
                                    scale=1
                                )
                        # Memory Optimization
                        with gr.Accordion("Model settings", open=True):
                            with gr.Row():
                                offload = gr.Checkbox(label="CPU Offloading", value=False, info="Offload models to CPU when not in use")
                                enable_fp8 = gr.Checkbox(label="FP8 Mode", value=False, info="Reduce memory with FP8 transformer")
                            gr.Markdown("### Block Swapping")
                            with gr.Row():
                                enable_dit_block_swap = gr.Checkbox(label="DiT Block Swap", value=True, info="Main transformer (stage 1)")
                                dit_blocks_in_memory = gr.Slider(minimum=1, maximum=47, value=22, step=1, label="DiT Blocks in GPU", visible=True)
                            with gr.Row():
                                enable_text_encoder_block_swap = gr.Checkbox(label="Text Encoder Block Swap", value=True, info="Gemma text encoder")
                                text_encoder_blocks_in_memory = gr.Slider(minimum=1, maximum=47, value=6, step=1, label="Text Encoder Blocks in GPU", visible=True, info="Gemma-3-12B has 48 layers")
                            with gr.Row():
                                enable_refiner_block_swap = gr.Checkbox(label="Refiner Block Swap", value=True, info="Stage 2 refiner transformer")
                                refiner_blocks_in_memory = gr.Slider(minimum=1, maximum=47, value=22, step=1, label="Refiner Blocks in GPU", visible=True)
                            with gr.Row():
                                ffn_chunk_size = gr.Slider(
                                    minimum=0, maximum=16384, value=0, step=512,
                                    label="FFN Chunk Size",
                                    info="Process FFN in chunks for long videos (0 = disabled, try 4096 for 1000+ frames)"
                                )
                            with gr.Row():
                                enable_activation_offload = gr.Checkbox(label="Activation Offload", value=False, info="Offload activations to CPU (slower but lower VRAM)")
                                temporal_chunk_size = gr.Slider(
                                    minimum=0, maximum=500000, value=0, step=50000,
                                    label="Temporal Chunk Size",
                                    info="Process video in chunks (0 = disabled, try 400000 for very long videos)"
                                )
                            with gr.Row():
                                disable_audio = gr.Checkbox(label="Disable Audio", value=False, info="Generate video only (no audio)")
                                enhance_prompt = gr.Checkbox(label="Enhance Prompt", value=False, info="Use Gemma to improve prompt")
                            with gr.Row():
                                checkpoint_path = gr.Textbox(
                                    label="LTX Checkpoint Path",
                                    value="./weights/ltx-2-19b-dev.safetensors",
                                    info="Path to LTX-2 model checkpoint",
                                    scale=4
                                )
                                distilled_checkpoint = gr.Checkbox(
                                    label="Distilled",
                                    value=False,
                                    info="Checkpoint is distilled (skips distilled LoRA)",
                                    scale=1
                                )
                            stage2_checkpoint = gr.Textbox(
                                label="Stage 2 Checkpoint (optional)",
                                value="",
                                info="Full model checkpoint for stage 2 refinement (leave empty to use distilled LoRA)",
                                placeholder="e.g., ./weights/ltx-2-19b-distilled.safetensors"
                            )
                            gemma_root = gr.Textbox(
                                label="Gemma Root Path",
                                value="./gemma-3-12b-it-qat-q4_0-unquantized",
                                info="Path to Gemma text encoder"
                            )
                            spatial_upsampler_path = gr.Textbox(
                                label="Spatial Upsampler Path",
                                value="./weights/ltx-2-spatial-upscaler-x2-1.0.safetensors",
                                info="Path to 2x spatial upsampler"
                            )
                            vae_path = gr.Textbox(
                                label="VAE Path (Optional)",
                                value="",
                                info="Path to dev checkpoint for vae use (leave empty to use VAE etc from main checkpoint)",
                                placeholder="e.g., ./home/mayble/h1111/ltx/weights/ltx-2-19b-dev.safetensors"
                            )
                            with gr.Row():
                                distilled_lora_path = gr.Textbox(
                                    label="Distilled LoRA Path",
                                    value="./weights/ltx-2-19b-distilled-lora-384.safetensors",
                                    info="For stage 2 refinement",
                                    scale=3
                                )
                                distilled_lora_strength = gr.Slider(
                                    minimum=0.0, maximum=2.0, value=1.0, step=0.1,
                                    label="Strength", scale=1
                                )
                            save_path = gr.Textbox(label="Output Folder", value="outputs")
                            with gr.Row():
                                lt1_save_defaults_btn = gr.Button("Save Defaults")
                                lt1_load_defaults_btn = gr.Button("Load Defaults")
                            lt1_defaults_status = gr.Textbox(label="Defaults Status", interactive=False, visible=False)
                        # Latent Normalization (fixes overbaking and audio clipping)
                        with gr.Accordion("Latent Normalization", open=False):
                            gr.Markdown("""
                            **Fixes overbaking and audio clipping** by normalizing latent values during denoising.
                            Apply stronger normalization early (high factors) and reduce later (low factors).
                            """)
                            latent_norm_mode = gr.Dropdown(
                                label="Normalization Mode",
                                choices=["none", "stat"],
                                value="none",
                                info="'none' = disabled, 'stat' = statistical normalization"
                            )
                            latent_norm_factors = gr.Textbox(
                                label="Per-Step Factors",
                                value="0.9,0.75,0.5,0.25,0.0",
                                info="Comma-separated factors for each step (higher = stronger normalization)"
                            )
                            with gr.Row():
                                latent_norm_target_mean = gr.Number(
                                    label="Target Mean",
                                    value=0.0,
                                    info="Target mean for normalization"
                                )
                                latent_norm_target_std = gr.Number(
                                    label="Target Std",
                                    value=1.0,
                                    info="Target standard deviation"
                                )
                            with gr.Row():
                                latent_norm_percentile = gr.Slider(
                                    minimum=50.0, maximum=100.0, value=95.0, step=1.0,
                                    label="Percentile",
                                    info="Percentile for outlier filtering (95 = ignore top/bottom 2.5%)"
                                )
                                latent_norm_clip_outliers = gr.Checkbox(
                                    label="Clip Outliers",
                                    value=False,
                                    info="Hard clip values outside percentile bounds"
                                )
                            with gr.Row():
                                latent_norm_video_only = gr.Checkbox(
                                    label="Video Only",
                                    value=False,
                                    info="Apply to video latents only"
                                )
                                latent_norm_audio_only = gr.Checkbox(
                                    label="Audio Only",
                                    value=False,
                                    info="Apply to audio latents only"
                                )

                        # V2A Mode (Video-to-Audio)
                        with gr.Accordion("V2A Mode (Video-to-Audio)", open=False):
                            gr.Markdown("""
                            **Generate audio for an existing silent video.**
                            The video frames are frozen and only audio is generated based on video content.
                            Upload your video in the **Video Input (V2V / Refine)** section above.
                            """)
                            v2a_mode = gr.Checkbox(
                                label="Enable V2A Mode",
                                value=False,
                                info="Freeze video, generate audio only (requires input video)"
                            )
                            v2a_strength = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                step=0.01,
                                value=1.0,
                                label="V2A Strength",
                                info="0.0 = keep original audio, 1.0 = full regeneration (only if input has audio)"
                            )

                        # Video Joining
                        with gr.Accordion("Video Joining", open=False):
                            gr.Markdown("""
                            **Join two videos with a generated AI transition.**
                            Upload two videos and the system will:
                            1. Find the sharpest frames near the transition points
                            2. Preserve sections from each video
                            3. Generate a smooth AI transition between them
                            """)

                            with gr.Row():
                                v2v_join_video1 = gr.Video(label="Video 1 (transition FROM end)", sources=["upload"])
                                v2v_join_video2 = gr.Video(label="Video 2 (transition TO start)", sources=["upload"])

                            with gr.Row():
                                v2v_join_frames_check1 = gr.Slider(
                                    minimum=1, maximum=120, value=30, step=1,
                                    label="Frames to Check (Video 1)",
                                    info="Number of frames from end of video1 to analyze for sharpest transition point"
                                )
                                v2v_join_frames_check2 = gr.Slider(
                                    minimum=1, maximum=120, value=30, step=1,
                                    label="Frames to Check (Video 2)",
                                    info="Number of frames from start of video2 to analyze for sharpest transition point"
                                )

                            with gr.Row():
                                v2v_join_preserve1 = gr.Slider(
                                    minimum=0.5, maximum=30.0, value=5.0, step=0.5,
                                    label="Preserve from Video 1 (seconds)",
                                    info="Seconds to preserve from end of video1"
                                )
                                v2v_join_preserve2 = gr.Slider(
                                    minimum=0.5, maximum=30.0, value=5.0, step=0.5,
                                    label="Preserve from Video 2 (seconds)",
                                    info="Seconds to preserve from start of video2"
                                )

                            with gr.Row():
                                v2v_join_transition_time = gr.Slider(
                                    minimum=1.0, maximum=30.0, value=10.0, step=0.5,
                                    label="Transition Duration (seconds)",
                                    info="Duration of AI-generated transition between preserved sections"
                                )

                            with gr.Row():
                                v2v_join_steps = gr.Slider(
                                    minimum=1, maximum=30, value=8, step=1,
                                    label="Denoising Steps",
                                    info="Number of denoising steps for transition generation"
                                )
                                v2v_join_terminal = gr.Slider(
                                    minimum=0.0, maximum=1.0, value=0.1, step=0.05,
                                    label="Terminal Sigma",
                                    info="Terminal sigma for partial denoising (lower = smoother)"
                                )

            # =================================================================
            # Video Info Tab
            # =================================================================
            with gr.Tab("Video Info"):
                with gr.Row():
                    info_video_input = gr.Video(label="Upload Video", interactive=True)
                    info_first_frame = gr.Image(label="First Frame", interactive=False)
                    info_metadata_output = gr.JSON(label="Generation Parameters")

                with gr.Row():
                    info_status = gr.Textbox(label="Status", interactive=False)

                with gr.Row():
                    info_send_to_v2v_btn = gr.Button("Send to V2V", variant="primary")
                    info_send_btn = gr.Button("Send to Generation", variant="secondary")
                    info_send_to_ext_btn = gr.Button("Send to Extension", variant="secondary")

            # =================================================================
            # SVI-LTX Tab
            # =================================================================
            with gr.Tab("SVI-LTX", id="svi_ltx_tab", visible=False):
                gr.Markdown("""
                ## SVI-LTX (Stable-Video-Infinity for LTX)
                Generate long, consistent videos by chaining multiple clips. Each clip uses motion latents from the previous clip for smooth transitions.
                **Features:** Multi-clip streaming • Motion latent conditioning • Per-clip prompts • Video extension
                """)

                # Clip Prompts - Two Column Layout per Accordion
                with gr.Accordion("Clip Prompts (1-4)", open=True):
                    with gr.Row():
                        with gr.Column():
                            svi_prompt1 = gr.Textbox(
                                label="Clip 1 Prompt (Required)", lines=2,
                                value="A fluffy cat and a curious rabbit discover a miniature fat man doll on the living room floor."
                            )
                            svi_prompt2 = gr.Textbox(
                                label="Clip 2 Prompt", lines=2,
                                value="The cat bats the tiny fat man doll with its paw while the rabbit hops around excitedly."
                            )
                        with gr.Column():
                            svi_prompt3 = gr.Textbox(
                                label="Clip 3 Prompt", lines=2,
                                value="The rabbit picks up the miniature fat man doll in its mouth and runs away, the cat chasing after it."
                            )
                            svi_prompt4 = gr.Textbox(
                                label="Clip 4 Prompt", lines=2,
                                value="The cat and rabbit sit together, both gazing at the tiny fat man doll between them."
                            )

                with gr.Accordion("Additional Clips (5-8)", open=False):
                    with gr.Row():
                        with gr.Column():
                            svi_prompt5 = gr.Textbox(label="Clip 5 Prompt", lines=2, value="")
                            svi_prompt6 = gr.Textbox(label="Clip 6 Prompt", lines=2, value="")
                        with gr.Column():
                            svi_prompt7 = gr.Textbox(label="Clip 7 Prompt", lines=2, value="")
                            svi_prompt8 = gr.Textbox(label="Clip 8 Prompt", lines=2, value="")

                with gr.Row():
                    with gr.Column(scale=4):
                        svi_negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            value="blurry, low quality, distorted, artifacts, ugly, deformed",
                            lines=2,
                        )
                    with gr.Column(scale=1):
                        svi_batch_size = gr.Number(label="Batch Count", value=1, minimum=1, step=1)
                    with gr.Column(scale=2):
                        svi_batch_progress = gr.Textbox(label="Status", interactive=False, value="")
                        svi_progress_text = gr.Textbox(label="Progress", interactive=False, value="")

                with gr.Row():
                    svi_generate_btn = gr.Button("🎬 Generate SVI Video", variant="primary", elem_classes="green-btn")
                    svi_stop_btn = gr.Button("⏹️ Stop", variant="stop")

                with gr.Row():
                    with gr.Column():
                        # Input images for SVI
                        with gr.Row():
                            svi_input_image = gr.Image(label="Input Image (Required)", type="filepath")
                            svi_anchor_image = gr.Image(label="Anchor Image (Optional)", type="filepath")
                        svi_original_dims = gr.Textbox(label="Original Dimensions", interactive=False, visible=False)

                        # Video Extension Mode
                        with gr.Accordion("Video Extension (Extend Existing Video)", open=False):
                            gr.Markdown("Extend an existing video by finding the best transition frame and generating new clips from it.")
                            svi_extend_video = gr.Video(label="Video to Extend", sources=["upload"])
                            with gr.Row():
                                svi_frames_to_check = gr.Slider(
                                    minimum=1, maximum=100, step=1, value=30,
                                    label="Frames to Check",
                                    info="Analyze last N frames to find sharpest transition point (1 = use last frame)"
                                )
                                svi_prepend_original = gr.Checkbox(
                                    label="Prepend Original",
                                    value=True,
                                    info="Include original video before extension"
                                )

                        # SVI-specific settings
                        gr.Markdown("### SVI Multi-Clip Settings")
                        with gr.Row():
                            svi_num_clips = gr.Slider(minimum=1, maximum=8, step=1, label="Number of Clips", value=4,
                                                     info="How many clips to chain together")
                            svi_overlap_frames = gr.Slider(minimum=0, maximum=16, step=1, label="Overlap Frames", value=1,
                                                          info="Overlapping frames between clips")
                            svi_num_motion_latent = gr.Slider(minimum=0, maximum=4, step=1, label="Motion Latent Frames", value=2,
                                                              info="Latent frames from previous clip for motion context")
                        with gr.Row():
                            svi_num_motion_frame = gr.Slider(minimum=1, maximum=8, step=1, label="Motion Frame Offset", value=1,
                                                              info="Frame offset from end for next clip input (1=last frame)")
                            svi_seed_multiplier = gr.Slider(minimum=1, maximum=200, step=1, label="Seed Multiplier", value=42,
                                                             info="Per-clip seed variation (seed = base + clip * multiplier)")

                        gr.Markdown("### Generation Parameters")
                        svi_scale_slider = gr.Slider(minimum=1, maximum=200, value=100, step=1, label="Scale %")
                        with gr.Row():
                            svi_width = gr.Number(label="Width", value=768, step=64, interactive=True)
                            svi_calc_height_btn = gr.Button("→")
                            svi_calc_width_btn = gr.Button("←")
                            svi_height = gr.Number(label="Height", value=512, step=64, interactive=True)
                        svi_num_frames = gr.Slider(minimum=9, maximum=241, step=8, label="Frames Per Clip", value=121, info="Frame count for each individual clip (8k+1)")
                        svi_frame_rate = gr.Slider(minimum=1, maximum=60, step=1, label="Frames Per Second", value=24)
                        svi_inference_steps = gr.Slider(minimum=4, maximum=100, step=1, label="Sampling Steps", value=40)
                        svi_cfg_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.1, label="Guidance Scale", value=4.0)
                        with gr.Row():
                            svi_seed = gr.Number(label="Seed (-1 for random)", value=-1)
                            svi_random_seed_btn = gr.Button("🎲")

                        with gr.Accordion("Anchor Settings", open=False):
                            gr.Markdown("Inject anchor image at regular intervals to guide video generation. small strength values seem to work best(.01-.05)")
                            svi_anchor_interval = gr.Number(
                                label="Anchor Interval",
                                value=0,
                                minimum=0,
                                step=8,
                                info="Frame interval (e.g., 60). Set to 0 to disable."
                            )
                            svi_anchor_strength = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.1, step=0.001,
                                label="Anchor Strength",
                                info="How strongly to guide toward anchor small values seem to work best(.01-.05)"
                            )
                            svi_anchor_decay = gr.Dropdown(
                                label="Anchor Decay",
                                choices=["none", "linear", "cosine", "sigmoid"],
                                value="cosine",
                                info="Decay schedule for anchor strength"
                            )

                    with gr.Column():
                        svi_output_gallery = gr.Gallery(
                            label="Generated Videos (Click to select)",
                            columns=[2], rows=[2], object_fit="contain", height="auto",
                            show_label=True, allow_preview=True, preview=True
                        )
                        with gr.Accordion("Preview (During Generation)", open=True):
                            svi_preview_gallery = gr.Gallery(
                                label="Clip Previews", columns=4, rows=2, object_fit="contain", height=300,
                                allow_preview=True, preview=True, show_label=True
                            )
                        with gr.Accordion("LoRA", open=True):
                            with gr.Row():
                                svi_lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                                svi_lora_refresh_btn = gr.Button("🔄 LoRA")
                            svi_lora_dropdown = gr.Dropdown(
                                label="Select LoRA",
                                choices=["None"],
                                value="None",
                                interactive=True
                            )
                            svi_lora_strength = gr.Slider(
                                minimum=0.0, maximum=2.0, step=0.05,
                                label="LoRA Strength", value=1.0
                            )

                # Model Settings
                with gr.Accordion("Model Settings", open=True):
                    with gr.Row():
                        svi_checkpoint_path = gr.Textbox(
                            label="LTX Checkpoint",
                            value="./weights/ltx-2-19b-dev.safetensors",
                            info="Main LTX model checkpoint"
                        )
                        svi_gemma_root = gr.Textbox(
                            label="Gemma Root",
                            value="./gemma-3-12b-it-qat-q4_0-unquantized",
                            info="Text encoder directory"
                        )
                    with gr.Row():
                        svi_spatial_upsampler = gr.Textbox(
                            label="Spatial Upsampler Path",
                            value="./weights/ltx-2-spatial-upscaler-x2-1.0.safetensors",
                            info="For 2x upscaling (two-stage)"
                        )
                        svi_distilled_lora = gr.Textbox(
                            label="Distilled LoRA Path",
                            value="./weights/ltx-2-19b-distilled-lora-384.safetensors",
                            info="For stage 2 refinement (two-stage)"
                        )
                    with gr.Row():
                        svi_distilled_lora_strength = gr.Slider(
                            minimum=0.0, maximum=2.0, step=0.05,
                            label="Distilled LoRA Strength", value=1.0
                        )
                    with gr.Row():
                        svi_one_stage = gr.Checkbox(label="One-Stage Pipeline", value=False, info="Skip two-stage refinement (faster but lower quality)")
                        svi_enable_fp8 = gr.Checkbox(label="Enable FP8", value=False, info="Use FP8 precision (lower VRAM)")
                        svi_offload = gr.Checkbox(label="Offload", value=False, info="Offload models to CPU")

                    # Block swap settings
                    gr.Markdown("### Block Swap Settings")
                    with gr.Row():
                        svi_enable_dit_block_swap = gr.Checkbox(label="DiT Block Swap", value=True)
                        svi_dit_blocks_in_memory = gr.Number(label="DiT Blocks in Memory", value=22, visible=True)
                    with gr.Row():
                        svi_enable_text_encoder_block_swap = gr.Checkbox(label="Text Encoder Block Swap", value=True)
                        svi_text_encoder_blocks_in_memory = gr.Number(label="Text Encoder Blocks in Memory", value=6, visible=True)
                    with gr.Row():
                        svi_enable_refiner_block_swap = gr.Checkbox(label="Refiner Block Swap", value=True)
                        svi_refiner_blocks_in_memory = gr.Number(label="Refiner Blocks in Memory", value=22, visible=True)
                    with gr.Row():
                        svi_ffn_chunk_size = gr.Slider(
                            minimum=0, maximum=16384, value=0, step=512,
                            label="FFN Chunk Size",
                            info="Process FFN in chunks for long videos (0 = disabled)"
                        )

                    with gr.Row():
                        svi_enable_activation_offload = gr.Checkbox(label="Activation Offload", value=False, info="Offload activations to CPU (slower but lower VRAM)")
                        svi_temporal_chunk_size = gr.Slider(
                            minimum=0, maximum=500000, value=0, step=50000,
                            label="Temporal Chunk Size",
                            info="Process video in chunks (0 = disabled)"
                        )
                    with gr.Row():
                        svi_disable_audio = gr.Checkbox(label="Disable Audio", value=False, info="Skip audio generation")
                    with gr.Row():
                        svi_output_path = gr.Textbox(
                            label="Output Path",
                            value="outputs",
                            info="Directory for generated videos"
                        )

                    # Preview Generation (SVI)
                    with gr.Accordion("Latent Preview (During Generation)", open=True):
                        svi_enable_preview = gr.Checkbox(label="Enable Latent Preview", value=True)
                        svi_preview_interval = gr.Slider(
                            minimum=1, maximum=50, step=1, value=5,
                            label="Preview Every N Steps"
                        )

                with gr.Row():
                    svi_save_defaults_btn = gr.Button("💾 Save Defaults")
                    svi_load_defaults_btn = gr.Button("📂 Load Defaults")

            # =================================================================
            # Depth Map Tab
            # =================================================================
            with gr.Tab("Depth Map", id="depth_tab"):
                gr.Markdown("""
                ## Depth Map Generator
                Generate depth maps from images or videos for use with LTX-2 IC-LoRA depth control.
                **Uses:** ZoeDepth (Intel/zoedepth-nyu-kitti) for monocular depth estimation.
                """)

                with gr.Row():
                    with gr.Column(scale=2):
                        # Input Section
                        with gr.Accordion("Input", open=True):
                            depth_input_type = gr.Radio(
                                label="Input Type",
                                choices=["Image", "Video"],
                                value="Image"
                            )
                            depth_input_image = gr.Image(
                                label="Input Image",
                                type="filepath",
                                visible=True
                            )
                            depth_input_video = gr.Video(
                                label="Input Video",
                                visible=False
                            )

                        # Output Settings
                        with gr.Accordion("Output Settings", open=True):
                            depth_output_type = gr.Radio(
                                label="Output Type",
                                choices=["Image", "Video"],
                                value="Image"
                            )
                            with gr.Row():
                                depth_colorize = gr.Checkbox(
                                    label="Colorize",
                                    value=False,
                                    info="Apply INFERNO colormap visualization"
                                )
                            with gr.Row():
                                depth_num_frames = gr.Number(
                                    label="Video Frames",
                                    value=121,
                                    visible=False,
                                    info="Number of frames for video output (Image→Video mode)"
                                )
                                depth_fps = gr.Number(
                                    label="FPS",
                                    value=24,
                                    visible=False
                                )
                            with gr.Row():
                                depth_max_frames = gr.Number(
                                    label="Max Frames",
                                    value=0,
                                    visible=False,
                                    info="Limit video frames (0 = all frames)"
                                )

                        # Resolution Settings
                        with gr.Accordion("Resolution (Optional)", open=False):
                            gr.Markdown("Leave at 0 to keep original resolution")
                            with gr.Row():
                                depth_width = gr.Number(label="Width", value=0, minimum=0, step=32)
                                depth_height = gr.Number(label="Height", value=0, minimum=0, step=32)

                        # Generate Button
                        with gr.Row():
                            depth_generate_btn = gr.Button("🗺️ Generate Depth Map", variant="primary")

                        # Status
                        depth_status = gr.Textbox(label="Status", interactive=False, value="Ready")

                    # Output Column
                    with gr.Column(scale=2):
                        depth_output_image = gr.Image(label="Depth Map Preview", visible=True)
                        depth_output_video = gr.Video(label="Depth Map Video", visible=False)

                        with gr.Row():
                            depth_send_to_gen_btn = gr.Button("📤 Send to Generation Tab", variant="secondary")

            # =================================================================
            # Frame Interpolation Tab (GIMM-VFI)
            # =================================================================
            with gr.Tab("Frame Interpolation", id="interp_tab"):
                gr.Markdown("### Increase Video FPS using GIMM-VFI\nState-of-the-art frame interpolation for smooth slow motion and higher frame rates.")

                with gr.Row():
                    # Input Column
                    with gr.Column(scale=1):
                        interp_input_video = gr.Video(label="Input Video", sources=["upload"])

                        # Model Settings
                        with gr.Accordion("Model Settings", open=True):
                            interp_model_variant = gr.Dropdown(
                                label="Model Variant",
                                choices=list(GIMM_MODELS.keys()),
                                value="GIMM-VFI-R-P (RAFT+Perceptual)",
                                info="R=RAFT (faster), F=FlowFormer (better quality), P=Perceptual loss (recommended)"
                            )
                            interp_checkpoint_path = gr.Textbox(
                                label="Checkpoint Path (optional)",
                                placeholder="Leave empty to use default for selected variant",
                                info="Override the default checkpoint path"
                            )
                            interp_config_path = gr.Textbox(
                                label="Config Path (optional)",
                                placeholder="Leave empty to use default for selected variant",
                                info="Override the default config path"
                            )

                        # Interpolation Settings
                        with gr.Accordion("Interpolation Settings", open=True):
                            interp_factor = gr.Slider(
                                label="Interpolation Factor",
                                minimum=2,
                                maximum=16,
                                value=2,
                                step=1,
                                info="2=2x FPS (1 new frame), 4=4x FPS (3 new frames), 8=8x FPS (7 new frames)"
                            )
                            interp_ds_scale = gr.Slider(
                                label="DS Scale (for high-res)",
                                minimum=0.25,
                                maximum=1.0,
                                value=1.0,
                                step=0.05,
                                info="Downscale factor: 1.0=SD/HD, 0.5=2K (~8GB VRAM), 0.25=4K (~11GB VRAM)"
                            )
                            interp_output_fps = gr.Number(
                                label="Output FPS Override",
                                value=0,
                                minimum=0,
                                info="0 = auto (input FPS × factor). Set manually for custom output FPS."
                            )

                        # Advanced Settings
                        with gr.Accordion("Advanced", open=False):
                            interp_raft_iters = gr.Slider(
                                label="RAFT Iterations (GIMM-VFI only)",
                                minimum=12,
                                maximum=32,
                                value=20,
                                step=1,
                                info="More iterations = better quality, slower (GIMM-VFI only)"
                            )
                            interp_pyr_level = gr.Slider(
                                label="Pyramid Level (BiM-VFI only)",
                                minimum=0,
                                maximum=8,
                                value=0,
                                step=1,
                                info="0=auto (based on resolution), 5=<1080p, 6=1080p, 7=4K+"
                            )
                            interp_seed = gr.Number(
                                label="Seed",
                                value=0,
                                info="Random seed for reproducibility"
                            )

                        # Upscaling Settings
                        with gr.Accordion("Upscaling", open=False):
                            upscale_enable = gr.Checkbox(
                                label="Enable Upscaling",
                                value=False,
                                info="Apply spatial upscaling (standalone or after interpolation)"
                            )
                            upscale_model = gr.Dropdown(
                                label="Upscaler Model",
                                choices=list(UPSCALER_MODELS.keys()),
                                value="Real-ESRGAN x2",
                                info="ESRGAN/SwinIR: frame-by-frame, BasicVSR++: temporal-aware"
                            )
                            upscale_tile_size = gr.Slider(
                                label="Tile Size",
                                minimum=0,
                                maximum=1024,
                                value=512,
                                step=64,
                                info="0=no tiling (more VRAM), 512=balanced, lower=less VRAM"
                            )
                            upscale_half = gr.Checkbox(
                                label="Half Precision (FP16)",
                                value=True,
                                info="Faster, less VRAM, slight quality loss"
                            )
                            upscale_model_path = gr.Textbox(
                                label="Custom Model Path (optional)",
                                placeholder="Leave empty for default model",
                                info="Override the default checkpoint path"
                            )
                            upscale_crf = gr.Slider(
                                label="Output CRF",
                                minimum=10,
                                maximum=30,
                                value=18,
                                step=1,
                                info="Video quality: lower=better quality, larger file (18=good default)"
                            )

                        # Motion Blur Settings (for masking deformation artifacts)
                        with gr.Accordion("Motion Blur (Artifact Masking)", open=False):
                            motion_blur_enable = gr.Checkbox(
                                label="Enable Motion Blur",
                                value=False,
                                info="Add blur along motion vectors to mask deformation artifacts"
                            )
                            motion_blur_strength = gr.Slider(
                                label="Blur Strength",
                                minimum=0.1,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                                info="Higher = more blur along motion direction"
                            )
                            motion_blur_samples = gr.Slider(
                                label="Blur Samples",
                                minimum=3,
                                maximum=15,
                                value=7,
                                step=2,
                                info="More samples = smoother blur (use odd numbers)"
                            )

                        # Action Buttons
                        with gr.Row():
                            interp_generate_btn = gr.Button("🎬 Interpolate", variant="primary", elem_classes="green-btn")
                            upscale_btn = gr.Button("🔍 Upscale", variant="secondary")

                    # Output Column
                    with gr.Column(scale=1):
                        interp_output_video = gr.Video(label="Interpolated Video")
                        interp_status = gr.Textbox(label="Status", value="Ready", interactive=False)
                        interp_progress = gr.Slider(
                            label="Progress",
                            minimum=0,
                            maximum=1,
                            value=0,
                            interactive=False,
                            visible=True
                        )

                        gr.Markdown("""
                        **Notes:**
                        - **GIMM-VFI**: Download from [HuggingFace](https://huggingface.co/GSean/GIMM-VFI). R=RAFT (faster), F=FlowFormer (better), P=Perceptual loss
                        - **BiM-VFI**: Download from [GitHub](https://github.com/KAIST-VICLab/BiM-VFI). Bidirectional motion field interpolation
                        - Place checkpoints in `GIMM-VFI/pretrained_ckpt/`
                        - For 2K/4K video with GIMM-VFI, reduce DS Scale to fit in VRAM
                        - BiM-VFI auto-detects pyramid level based on resolution (or set manually)

                        **Upscaling:**
                        - **Real-ESRGAN**: Download from [GitHub](https://github.com/xinntao/Real-ESRGAN/releases)
                        - **SwinIR**: Download from [GitHub](https://github.com/JingyunLiang/SwinIR/releases)
                        - Place upscaler checkpoints in `weights/` folder
                        - Motion blur uses RAFT flow to mask deformation artifacts
                        """)

            # =================================================================
            # Extension Tab (Wan2GP-style)
            # =================================================================
            with gr.Tab("Extension", id="ext_tab", visible=False):
                with gr.Row():
                    # Left column - Input and settings
                    with gr.Column(scale=1):
                        # Extension-specific inputs
                        with gr.Accordion("Input Video", open=True):
                            ext_input_video = gr.Video(label="Video to Extend", sources=["upload"])
                            ext_prompt = gr.Textbox(
                                label="Prompt",
                                placeholder="Describe the continuation...",
                                lines=3
                            )
                            ext_negative_prompt = gr.Textbox(
                                label="Negative Prompt",
                                placeholder="Things to avoid...",
                                lines=2
                            )

                        with gr.Accordion("Extension Settings", open=True):
                            with gr.Row():
                                ext_extend_seconds = gr.Slider(
                                    minimum=1.0, maximum=30.0, value=5.0, step=0.5,
                                    label="Extend Duration (seconds)"
                                )
                                ext_preserve_seconds = gr.Slider(
                                    minimum=0.0, maximum=30.0, value=0.0, step=0.5,
                                    label="Max Preserve (seconds)",
                                    info="0 = keep all, >0 = only use last N seconds as context"
                                )
                            with gr.Row():
                                ext_seed = gr.Number(label="Seed", value=-1, precision=0)
                                ext_random_seed_btn = gr.Button("🎲", size="sm")
                            with gr.Row():
                                ext_steps = gr.Slider(
                                    minimum=4, maximum=60, value=30, step=1,
                                    label="Inference Steps"
                                )
                                ext_cfg = gr.Slider(
                                    minimum=1.0, maximum=10.0, value=3.0, step=0.1,
                                    label="CFG Scale"
                                )
                            with gr.Row():
                                ext_preserve_strength = gr.Slider(
                                    minimum=0.5, maximum=1.0, value=1.0, step=0.05,
                                    label="Preserve Strength",
                                    info="1.0 = fully frozen original frames"
                                )
                                ext_skip_stage2 = gr.Checkbox(
                                    label="Skip Stage 2",
                                    value=False,
                                    info="Faster but lower quality"
                                )
                            with gr.Row():
                                ext_batch_size = gr.Number(
                                    label="Batch Count", value=1, minimum=1, step=1,
                                    info="Number of videos to generate"
                                )

                    # Right column - Output, LoRAs and Model Settings
                    with gr.Column(scale=1):
                        # Output gallery and controls
                        ext_output_gallery = gr.Gallery(
                            label="Extended Videos",
                            columns=1, rows=1,
                            object_fit="contain",
                            height="auto",
                            allow_preview=True,
                            preview=True
                        )
                        with gr.Row():
                            ext_generate_btn = gr.Button("Extend Video", variant="primary", size="lg")
                            ext_stop_btn = gr.Button("Stop", variant="stop", size="lg")
                        ext_status_text = gr.Textbox(label="Status", value="Ready", interactive=False)
                        ext_progress_text = gr.Textbox(label="Progress", value="", interactive=False)

                        # User LoRAs (copied from main tab)
                        with gr.Accordion("User LoRAs (Optional)", open=True):
                            ext_lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                            ext_lora_refresh_btn = gr.Button("🔄 Refresh", size="sm")
                            # LoRA 1
                            with gr.Row():
                                ext_user_lora_1 = gr.Dropdown(
                                    label="LoRA 1",
                                    choices=get_ltx_lora_options("lora"),
                                    value="None",
                                    scale=3
                                )
                                ext_user_lora_strength_1 = gr.Slider(
                                    minimum=0.0, maximum=2.0, value=1.0, step=0.1,
                                    label="Strength", scale=1
                                )
                                ext_user_lora_stage_1 = gr.Dropdown(
                                    label="Stage",
                                    choices=["Stage 1 (Base)", "Stage 2 (Refine)", "Stage 3 (Refine)", "All"],
                                    value="Stage 2 (Refine)",
                                    scale=1
                                )
                            # LoRA 2
                            with gr.Row():
                                ext_user_lora_2 = gr.Dropdown(
                                    label="LoRA 2",
                                    choices=get_ltx_lora_options("lora"),
                                    value="None",
                                    scale=3
                                )
                                ext_user_lora_strength_2 = gr.Slider(
                                    minimum=0.0, maximum=2.0, value=1.0, step=0.1,
                                    label="Strength", scale=1
                                )
                                ext_user_lora_stage_2 = gr.Dropdown(
                                    label="Stage",
                                    choices=["Stage 1 (Base)", "Stage 2 (Refine)", "Stage 3 (Refine)", "All"],
                                    value="Stage 2 (Refine)",
                                    scale=1
                                )
                            # LoRA 3
                            with gr.Row():
                                ext_user_lora_3 = gr.Dropdown(
                                    label="LoRA 3",
                                    choices=get_ltx_lora_options("lora"),
                                    value="None",
                                    scale=3
                                )
                                ext_user_lora_strength_3 = gr.Slider(
                                    minimum=0.0, maximum=2.0, value=1.0, step=0.1,
                                    label="Strength", scale=1
                                )
                                ext_user_lora_stage_3 = gr.Dropdown(
                                    label="Stage",
                                    choices=["Stage 1 (Base)", "Stage 2 (Refine)", "Stage 3 (Refine)", "All"],
                                    value="Stage 2 (Refine)",
                                    scale=1
                                )
                            # LoRA 4
                            with gr.Row():
                                ext_user_lora_4 = gr.Dropdown(
                                    label="LoRA 4",
                                    choices=get_ltx_lora_options("lora"),
                                    value="None",
                                    scale=3
                                )
                                ext_user_lora_strength_4 = gr.Slider(
                                    minimum=0.0, maximum=2.0, value=1.0, step=0.1,
                                    label="Strength", scale=1
                                )
                                ext_user_lora_stage_4 = gr.Dropdown(
                                    label="Stage",
                                    choices=["Stage 1 (Base)", "Stage 2 (Refine)", "Stage 3 (Refine)", "All"],
                                    value="Stage 2 (Refine)",
                                    scale=1
                                )

                        # Model settings (copied from main tab)
                        with gr.Accordion("Model Settings", open=True):
                            with gr.Row():
                                ext_offload = gr.Checkbox(label="CPU Offloading", value=False, info="Offload models to CPU when not in use")
                                ext_enable_fp8 = gr.Checkbox(label="FP8 Mode", value=False, info="Reduce memory with FP8 transformer")
                            gr.Markdown("### Block Swapping")
                            with gr.Row():
                                ext_enable_dit_block_swap = gr.Checkbox(label="DiT Block Swap", value=True, info="Main transformer (stage 1)")
                                ext_dit_blocks_in_memory = gr.Slider(minimum=1, maximum=47, value=22, step=1, label="DiT Blocks in GPU", visible=True)
                            with gr.Row():
                                ext_enable_text_encoder_block_swap = gr.Checkbox(label="Text Encoder Block Swap", value=True, info="Gemma text encoder")
                                ext_text_encoder_blocks_in_memory = gr.Slider(minimum=1, maximum=47, value=6, step=1, label="Text Encoder Blocks in GPU", visible=True, info="Gemma-3-12B has 48 layers")
                            with gr.Row():
                                ext_enable_refiner_block_swap = gr.Checkbox(label="Refiner Block Swap", value=True, info="Stage 2 refiner transformer")
                                ext_refiner_blocks_in_memory = gr.Slider(minimum=1, maximum=47, value=22, step=1, label="Refiner Blocks in GPU", visible=True)
                            with gr.Row():
                                ext_enable_activation_offload = gr.Checkbox(label="Activation Offload", value=False, info="Offload activations to CPU (slower but lower VRAM)")
                            with gr.Row():
                                ext_checkpoint_path = gr.Textbox(
                                    label="LTX Checkpoint Path",
                                    value="./weights/ltx-2-19b-dev.safetensors",
                                    info="Path to LTX-2 model checkpoint",
                                    scale=4
                                )
                                ext_distilled_checkpoint = gr.Checkbox(
                                    label="Distilled",
                                    value=False,
                                    info="Checkpoint is distilled (skips distilled LoRA)",
                                    scale=1
                                )
                            ext_gemma_root = gr.Textbox(
                                label="Gemma Root Path",
                                value="./gemma-3-12b-it-qat-q4_0-unquantized",
                                info="Path to Gemma text encoder"
                            )
                            ext_spatial_upsampler_path = gr.Textbox(
                                label="Spatial Upsampler Path",
                                value="./weights/ltx-2-spatial-upscaler-x2-1.0.safetensors",
                                info="Path to 2x spatial upsampler"
                            )
                            ext_vae_path = gr.Textbox(
                                label="VAE Path (Optional)",
                                value="",
                                info="Path to dev checkpoint for vae use (leave empty to use VAE from main checkpoint)",
                                placeholder="e.g., ./weights/ltx-2-19b-dev.safetensors"
                            )
                            with gr.Row():
                                ext_distilled_lora_path = gr.Textbox(
                                    label="Distilled LoRA Path",
                                    value="./weights/ltx-2-19b-distilled-lora-384.safetensors",
                                    info="For stage 2 refinement",
                                    scale=3
                                )
                                ext_distilled_lora_strength = gr.Slider(
                                    minimum=0.0, maximum=2.0, value=1.0, step=0.1,
                                    label="Strength", scale=1
                                )
                            ext_save_path = gr.Textbox(label="Output Folder", value="outputs")
                            with gr.Row():
                                ext_save_defaults_btn = gr.Button("💾 Save Defaults")
                                ext_load_defaults_btn = gr.Button("📂 Load Defaults")
                            ext_defaults_status = gr.Textbox(label="Defaults Status", value="", interactive=False, visible=True)

                # Tips at the bottom (outside columns)
                gr.Markdown("""
                ---
                **How it works:**
                - Uses VideoConditionByLatentIndex to preserve original frames exactly
                - Generates seamless continuation with synchronized audio
                - Stage 1: Low-res extension at half resolution
                - Stage 2: Hi-res refinement with 2x upsampling

                **Tips:**
                - Use a descriptive prompt for the continuation
                - Preserve Strength 1.0 keeps original frames frozen
                - Skip Stage 2 for faster (but lower quality) results
                - **Max Preserve**: Set to limit VRAM usage by only using the last N seconds as context
                  - 0 = use entire video as context (may OOM on long videos)
                  - \>0 = only encode last N seconds, full video still included in output
                """)

            # =================================================================
            # Help Tab
            # =================================================================
            with gr.Tab("Help"):
                gr.Markdown("""
                ## LTX-2 Video Generation Help

                ### Required Model Files
                **For two-stage pipeline** (default, higher quality):
                1. **LTX Checkpoint** - Main 19B model (.safetensors)
                2. **Gemma Root** - Text encoder directory
                3. **Spatial Upsampler** - For 2x resolution upscaling
                4. **Distilled LoRA** - For stage 2 refinement

                **For one-stage pipeline** (faster):
                1. **LTX Checkpoint** - Main 19B model (.safetensors)
                2. **Gemma Root** - Text encoder directory

                ### Pipeline Options
                - **Two-stage** (default): Generates at half resolution, upsamples 2x, then refines with distilled LoRA. Higher quality output.
                - **One-stage**: Generates directly at full resolution in a single pass. Faster but may have less detail.
                - **Refine-only**: Runs stage 2 refinement on an input video. Use this to improve a previously generated video.

                ### Stage 2 Checkpoint
                - By default, stage 2 uses the main checkpoint + distilled LoRA for refinement
                - **Stage 2 Checkpoint** (optional): Use a separate full model checkpoint for stage 2 (e.g., a distilled model)
                - When a stage 2 checkpoint is provided, the distilled LoRA is not applied

                ### Generation Modes
                - **T2V (Text-to-Video)**: Generate video from text prompt only
                - **I2V (Image-to-Video)**: Start from an input image
                - **V2V (Video-to-Video)**: Refine an existing video

                ### Video Input (V2V / Refine)
                - Upload a video to refine it using stage 2
                - Use "refine-only" pipeline to skip stage 1 generation
                - **Refine Strength**: Amount of noise added before refinement (0=none, 1=full denoise)
                - Distilled LoRA is optional for refine-only mode

                ### Image Conditioning (I2V)
                - **Start Image**: Set the first frame of the video
                - **End Image** (optional): Set the last frame to create start-to-end transitions
                - **Scale Slider**: Automatically adjusts width/height based on input image aspect ratio
                - **→ / ← Buttons**: Calculate width from height or height from width while preserving aspect ratio

                ### Resolution Requirements
                - Width and height must be divisible by 64
                - Recommended: 1024x1024, 768x1024, 1024x768
                - When using I2V, the scale slider helps maintain the input image's aspect ratio

                ### Frame Count
                - Must be in format: 8*K + 1
                - Examples: 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121
                - 121 frames @ 24fps = 5 seconds

                ### Memory Optimization
                - **CPU Offloading**: Reduces peak VRAM by moving models to CPU
                - **FP8 Mode**: Stores transformer in FP8 (lower precision)
                - **Block Swapping**: Keeps only N transformer blocks in GPU (~40% VRAM reduction)

                ### Audio Generation
                - LTX-2 generates synchronized audio by default
                - Disable audio if you only need video
                """)

        # =================================================================
        # Event Handlers
        # =================================================================

        # Random seed button
        random_seed_btn.click(
            fn=lambda: -1,
            outputs=[seed]
        )

        # LoRA refresh (updates all 4 dropdowns)
        def refresh_all_lora_dropdowns(folder):
            choices = get_ltx_lora_options(folder)
            return [gr.update(choices=choices) for _ in range(4)]

        lora_refresh_btn.click(
            fn=refresh_all_lora_dropdowns,
            inputs=[lora_folder],
            outputs=[user_lora_1, user_lora_2, user_lora_3, user_lora_4]
        )

        # Block swap visibility toggles
        enable_dit_block_swap.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[enable_dit_block_swap],
            outputs=[dit_blocks_in_memory]
        )
        enable_text_encoder_block_swap.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[enable_text_encoder_block_swap],
            outputs=[text_encoder_blocks_in_memory]
        )
        enable_refiner_block_swap.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[enable_refiner_block_swap],
            outputs=[refiner_blocks_in_memory]
        )

        # Image dimension handlers
        input_image.change(
            fn=update_image_dimensions,
            inputs=[input_image],
            outputs=[original_dims, width, height]
        )

        scale_slider.change(
            fn=update_resolution_from_scale,
            inputs=[scale_slider, original_dims],
            outputs=[width, height]
        )

        calc_width_btn.click(
            fn=calculate_width_from_height,
            inputs=[height, original_dims],
            outputs=[width]
        )

        calc_height_btn.click(
            fn=calculate_height_from_width,
            inputs=[width, original_dims],
            outputs=[height]
        )

        # Mode change - show/hide I2V and V2V sections
        def update_sections_visibility(m):
            return gr.update(open=(m == "i2v")), gr.update(open=(m == "v2v"))

        mode.change(
            fn=update_sections_visibility,
            inputs=[mode],
            outputs=[i2v_section, v2v_section]
        )

        # V2V Audio Mode change - show/hide V2V Audio Strength slider
        v2v_audio_mode.change(
            fn=lambda m: gr.update(visible=(m == "condition")),
            inputs=[v2v_audio_mode],
            outputs=[v2v_audio_strength]
        )

        # Video input change - update dimensions and frame count
        input_video.change(
            fn=update_video_dimensions,
            inputs=[input_video],
            outputs=[original_dims, width, height, num_frames, frame_rate]
        )

        # Stop button
        stop_btn.click(
            fn=stop_generation,
            outputs=[status_text]
        )

        # Generate button
        generate_btn.click(
            fn=generate_ltx_video,
            inputs=[
                prompt, negative_prompt,
                checkpoint_path, distilled_checkpoint, stage2_checkpoint, gemma_root, spatial_upsampler_path,
                vae_path, distilled_lora_path, distilled_lora_strength,
                mode, pipeline, sampler, stage2_sampler, enable_sliding_window, width, height, num_frames, frame_rate,
                cfg_guidance_scale, num_inference_steps, stage2_steps, stage2_strength, stage3_strength, seed,
                stg_scale, stg_blocks, stg_mode,
                # Advanced CFG (MultiModal Guidance)
                guidance_mode, scheduler_type,
                video_cfg_guidance_scale, video_stg_scale, video_stg_blocks, video_rescale_scale,
                a2v_guidance_scale, video_skip_step,
                audio_cfg_guidance_scale, audio_stg_scale, audio_stg_blocks, audio_rescale_scale,
                v2a_guidance_scale, audio_skip_step,
                kandinsky_scheduler, kandinsky_scheduler_scale,
                input_image, image_frame_idx, image_strength, image_crf,
                end_image, end_image_strength, end_image_crf,
                anchor_image, anchor_interval, anchor_strength, anchor_decay, anchor_crf,
                input_video, refine_strength,
                disable_audio, audio_input, audio_strength, v2v_audio_mode, v2v_audio_strength, enhance_prompt,
                offload, enable_fp8,
                enable_dit_block_swap, dit_blocks_in_memory,
                enable_text_encoder_block_swap, text_encoder_blocks_in_memory,
                enable_refiner_block_swap, refiner_blocks_in_memory,
                ffn_chunk_size, enable_activation_offload, temporal_chunk_size,
                lora_folder,
                user_lora_1, user_lora_strength_1, user_lora_stage_1,
                user_lora_2, user_lora_strength_2, user_lora_stage_2,
                user_lora_3, user_lora_strength_3, user_lora_stage_3,
                user_lora_4, user_lora_strength_4, user_lora_stage_4,
                save_path, batch_size,
                # Preview Generation
                enable_preview, preview_interval,
                # Video Continuation (Frame Freezing)
                freeze_frames, freeze_transition,
                # Sliding Window (Long Video)
                sliding_window_size, sliding_window_overlap,
                sliding_window_overlap_noise, sliding_window_color_correction,
                # AV Extension (Time-Based Audio-Video Continuation)
                av_extend_video, av_extend_start_time, av_extend_end_time,
                av_extend_steps, av_extend_terminal, av_slope_len, av_no_stage2,
                # Depth Control (IC-LoRA)
                depth_control_video, depth_control_image, estimate_depth,
                depth_strength, depth_stage2,
                # Latent Normalization
                latent_norm_mode, latent_norm_factors,
                latent_norm_target_mean, latent_norm_target_std,
                latent_norm_percentile, latent_norm_clip_outliers,
                latent_norm_video_only, latent_norm_audio_only,
                # V2A Mode
                v2a_mode, v2a_strength,
                # Video Joining
                v2v_join_video1, v2v_join_video2,
                v2v_join_frames_check1, v2v_join_frames_check2,
                v2v_join_preserve1, v2v_join_preserve2,
                v2v_join_transition_time,
                v2v_join_steps, v2v_join_terminal,
            ],
            outputs=[output_gallery, preview_gallery, status_text, progress_text]
        )

        # =================================================================
        # Video Info Tab Event Handlers
        # =================================================================
        info_video_input.upload(
            fn=extract_video_details,
            inputs=info_video_input,
            outputs=[info_metadata_output, info_status, info_first_frame]
        )

        def send_to_generation_handler(metadata, first_frame):
            """Send loaded metadata to generation tab parameters and switch to Generation tab."""
            if not metadata:
                return [gr.update()] * 58 + ["No metadata loaded - upload a video first"]

            # Handle legacy metadata that used single enable_block_swap
            legacy_block_swap = metadata.get("enable_block_swap", True)

            # Extract image conditioning info from metadata
            # Image tuple format: (path, frame_idx, strength, crf) - crf may be missing in older metadata
            images = metadata.get("images", [])
            image_strength = 0.9
            image_frame_idx = 0
            image_crf = 33
            end_image_strength = 0.9
            end_image_crf = 33

            if images and len(images) > 0:
                # First image entry (start image): (path, frame_idx, strength, crf)
                image_frame_idx = images[0][1] if len(images[0]) > 1 else 0
                image_strength = images[0][2] if len(images[0]) > 2 else 0.9
                image_crf = images[0][3] if len(images[0]) > 3 else 33

                # Check for end image (second entry, typically at last frame)
                if len(images) > 1:
                    end_image_strength = images[1][2] if len(images[1]) > 2 else 0.9
                    end_image_crf = images[1][3] if len(images[1]) > 3 else 33

            # Determine mode based on whether images were used
            mode = "t2v"
            if images:
                mode = "i2v"
            elif metadata.get("input_video"):
                mode = "v2v"

            # Return updates for all generation parameters
            # NOTE: Model paths (checkpoint_path, gemma_root, spatial_upsampler_path,
            #       distilled_lora_path) are NOT restored - user keeps their current settings
            return [
                gr.Tabs(selected="gen_tab"),  # Switch to Generation tab
                gr.update(value=metadata.get("prompt", "")),  # prompt
                gr.update(value=metadata.get("negative_prompt", "")),  # negative_prompt
                gr.update(value=mode),  # mode
                gr.update(value=metadata.get("pipeline", "two-stage")),  # pipeline
                gr.update(value=metadata.get("sampler", "euler")),  # sampler
                gr.update(value=metadata.get("stage2_sampler", "euler")),  # stage2_sampler
                gr.update(value=metadata.get("width", 1024)),  # width
                gr.update(value=metadata.get("height", 1024)),  # height
                gr.update(value=metadata.get("num_frames", 121)),  # num_frames
                gr.update(value=metadata.get("frame_rate", 24)),  # frame_rate
                gr.update(value=metadata.get("cfg_guidance_scale", 4.0)),  # cfg_guidance_scale
                gr.update(value=metadata.get("num_inference_steps", 40)),  # num_inference_steps
                gr.update(value=metadata.get("stage2_steps", 3)),  # stage2_steps
                gr.update(value=metadata.get("stage2_strength", 1.0)),  # stage2_strength
                gr.update(value=metadata.get("stage3_strength", 1.0)),  # stage3_strength
                gr.update(value=metadata.get("seed", -1)),  # seed
                # STG parameters
                gr.update(value=metadata.get("stg_scale", 0.0)),  # stg_scale
                gr.update(value=metadata.get("stg_blocks", "29")),  # stg_blocks
                gr.update(value=metadata.get("stg_mode", "stg_av") or "stg_av"),  # stg_mode
                # Advanced CFG (MultiModal Guidance)
                gr.update(value=metadata.get("guidance_mode", "legacy")),  # guidance_mode
                gr.update(value=metadata.get("scheduler_type", "ltx2")),  # scheduler_type
                gr.update(value=metadata.get("video_cfg_guidance_scale", 3.0)),  # video_cfg_guidance_scale
                gr.update(value=metadata.get("video_stg_scale", 1.0)),  # video_stg_scale
                gr.update(value=metadata.get("video_stg_blocks", "29")),  # video_stg_blocks
                gr.update(value=metadata.get("video_rescale_scale", 0.7)),  # video_rescale_scale
                gr.update(value=metadata.get("a2v_guidance_scale", 3.0)),  # a2v_guidance_scale
                gr.update(value=metadata.get("video_skip_step", 0)),  # video_skip_step
                gr.update(value=metadata.get("audio_cfg_guidance_scale", 7.0)),  # audio_cfg_guidance_scale
                gr.update(value=metadata.get("audio_stg_scale", 1.0)),  # audio_stg_scale
                gr.update(value=metadata.get("audio_stg_blocks", "29")),  # audio_stg_blocks
                gr.update(value=metadata.get("audio_rescale_scale", 0.7)),  # audio_rescale_scale
                gr.update(value=metadata.get("v2a_guidance_scale", 3.0)),  # v2a_guidance_scale
                gr.update(value=metadata.get("audio_skip_step", 0)),  # audio_skip_step
                # Kandinsky scheduler parameters
                gr.update(value=metadata.get("kandinsky_scheduler", False)),  # kandinsky_scheduler
                gr.update(value=metadata.get("kandinsky_scheduler_scale", 5.0)),  # kandinsky_scheduler_scale
                # Image conditioning (extracted from images tuple)
                gr.update(value=first_frame),  # input_image - use extracted first frame
                gr.update(value=image_frame_idx),  # image_frame_idx
                gr.update(value=image_strength),  # image_strength
                gr.update(value=image_crf),  # image_crf
                gr.update(value=end_image_strength),  # end_image_strength
                gr.update(value=end_image_crf),  # end_image_crf
                # Anchor conditioning
                gr.update(value=metadata.get("anchor_interval", 0) or 0),  # anchor_interval
                gr.update(value=metadata.get("anchor_strength", 0.8)),  # anchor_strength
                gr.update(value=metadata.get("anchor_decay", "cosine") or "cosine"),  # anchor_decay
                gr.update(value=metadata.get("anchor_crf", 33)),  # anchor_crf
                # Refine settings
                gr.update(value=metadata.get("refine_strength", 0.3)),  # refine_strength
                # Audio and prompt
                gr.update(value=metadata.get("disable_audio", False)),  # disable_audio
                gr.update(value=metadata.get("audio_strength", 1.0)),  # audio_strength
                gr.update(value=metadata.get("v2v_audio_mode", "preserve")),  # v2v_audio_mode
                gr.update(value=metadata.get("v2v_audio_strength", 1.0)),  # v2v_audio_strength
                gr.update(value=metadata.get("enhance_prompt", False)),  # enhance_prompt
                # Memory optimization
                gr.update(value=metadata.get("offload", False)),  # offload
                gr.update(value=metadata.get("enable_fp8", False)),  # enable_fp8
                # Block swap settings
                gr.update(value=metadata.get("enable_dit_block_swap", legacy_block_swap)),  # enable_dit_block_swap
                gr.update(value=metadata.get("dit_blocks_in_memory", 22) or 22),  # dit_blocks_in_memory
                gr.update(value=metadata.get("enable_text_encoder_block_swap", legacy_block_swap)),  # enable_text_encoder_block_swap
                gr.update(value=metadata.get("text_encoder_blocks_in_memory", 6) or 6),  # text_encoder_blocks_in_memory
                gr.update(value=metadata.get("enable_refiner_block_swap", legacy_block_swap)),  # enable_refiner_block_swap
                gr.update(value=metadata.get("refiner_blocks_in_memory", 22) or 22),  # refiner_blocks_in_memory
                # Distilled settings (NOT model paths)
                gr.update(value=metadata.get("distilled_checkpoint", False)),  # distilled_checkpoint
                # NOTE: stage2_checkpoint path is NOT restored - keep user's current setting
                "Parameters sent to Generation tab (model paths unchanged)"  # status
            ]

        def send_to_v2v_handler(metadata, video_path):
            """Send loaded video to V2V input in generation tab and switch to Generation tab."""
            if not video_path:
                return [gr.update()] * 8 + ["No video loaded - upload a video first"]

            # Return updates for V2V mode
            return [
                gr.Tabs(selected="gen_tab"),  # Switch to Generation tab
                gr.update(value=video_path),  # input_video - send to V2V input
                gr.update(value="v2v"),  # mode - set to v2v
                gr.update(value=metadata.get("prompt", "") if metadata else ""),  # prompt
                gr.update(value=metadata.get("negative_prompt", "") if metadata else ""),  # negative_prompt
                gr.update(value=metadata.get("seed", -1) if metadata else -1),  # seed
                gr.update(value=metadata.get("refine_strength", 0.3) if metadata else 0.3),  # refine_strength
                gr.update(value="preserve"),  # v2v_audio_mode - default to preserve
                "Video sent to V2V input in Generation tab"  # status
            ]

        def send_to_extension_handler(metadata, video_path):
            """Send loaded video to Extension tab and switch to Extension tab."""
            if not video_path:
                return [gr.update()] * 5 + ["No video loaded - upload a video first"]

            # Return updates for Extension tab parameters
            return [
                gr.Tabs(selected="ext_tab"),  # Switch to Extension tab
                gr.update(value=video_path),  # ext_input_video
                gr.update(value=metadata.get("prompt", "") if metadata else ""),  # ext_prompt
                gr.update(value=metadata.get("negative_prompt", "") if metadata else ""),  # ext_negative_prompt
                gr.update(value=metadata.get("seed", -1) if metadata else -1),  # ext_seed
                "Video sent to Extension tab"  # status
            ]

        info_send_btn.click(
            fn=send_to_generation_handler,
            inputs=[info_metadata_output, info_first_frame],
            outputs=[
                tabs,  # Switch tab
                prompt, negative_prompt, mode, pipeline, sampler, stage2_sampler,
                width, height, num_frames, frame_rate,
                cfg_guidance_scale, num_inference_steps, stage2_steps, stage2_strength, stage3_strength, seed,
                # STG parameters
                stg_scale, stg_blocks, stg_mode,
                # Advanced CFG (MultiModal Guidance)
                guidance_mode, scheduler_type,
                video_cfg_guidance_scale, video_stg_scale, video_stg_blocks, video_rescale_scale,
                a2v_guidance_scale, video_skip_step,
                audio_cfg_guidance_scale, audio_stg_scale, audio_stg_blocks, audio_rescale_scale,
                v2a_guidance_scale, audio_skip_step,
                # Kandinsky scheduler parameters
                kandinsky_scheduler, kandinsky_scheduler_scale,
                # Image conditioning
                input_image, image_frame_idx, image_strength, image_crf, end_image_strength, end_image_crf,
                # Anchor conditioning
                anchor_interval, anchor_strength, anchor_decay, anchor_crf,
                # Refine settings
                refine_strength,
                # Audio and prompt
                disable_audio, audio_strength, v2v_audio_mode, v2v_audio_strength, enhance_prompt,
                # Memory optimization
                offload, enable_fp8,
                # Block swap settings
                enable_dit_block_swap, dit_blocks_in_memory,
                enable_text_encoder_block_swap, text_encoder_blocks_in_memory,
                enable_refiner_block_swap, refiner_blocks_in_memory,
                # Distilled settings
                distilled_checkpoint,
                info_status
            ]
        )

        info_send_to_v2v_btn.click(
            fn=send_to_v2v_handler,
            inputs=[info_metadata_output, info_video_input],
            outputs=[
                tabs,  # Switch tab
                input_video,  # V2V input video
                mode,  # Set mode to v2v
                prompt,  # Prompt
                negative_prompt,  # Negative prompt
                seed,  # Seed
                refine_strength,  # Refine strength
                v2v_audio_mode,  # Audio mode (preserve)
                info_status  # Status
            ]
        )

        info_send_to_ext_btn.click(
            fn=send_to_extension_handler,
            inputs=[info_metadata_output, info_video_input],
            outputs=[
                tabs,  # Switch tab
                ext_input_video,  # Video to extend
                ext_prompt,  # Prompt
                ext_negative_prompt,  # Negative prompt
                ext_seed,  # Seed
                info_status  # Status
            ]
        )

        # =================================================================
        # SVI-LTX Event Handlers
        # =================================================================

        # SVI-LTX Generate button
        svi_generate_btn.click(
            fn=generate_svi_ltx_video,
            inputs=[
                # Prompts (8)
                svi_prompt1, svi_prompt2, svi_prompt3, svi_prompt4,
                svi_prompt5, svi_prompt6, svi_prompt7, svi_prompt8,
                svi_negative_prompt,
                # Images
                svi_input_image, svi_anchor_image,
                # Video extension
                svi_extend_video, svi_frames_to_check, svi_prepend_original,
                # SVI settings
                svi_num_clips, svi_overlap_frames, svi_num_motion_latent,
                svi_num_motion_frame, svi_seed_multiplier,
                # Generation params
                svi_width, svi_height, svi_num_frames, svi_frame_rate,
                svi_inference_steps, svi_cfg_scale, svi_seed,
                # Anchor settings
                svi_anchor_interval, svi_anchor_strength, svi_anchor_decay,
                # Model settings
                svi_checkpoint_path, svi_gemma_root, svi_spatial_upsampler,
                svi_distilled_lora, svi_distilled_lora_strength, svi_one_stage,
                svi_enable_fp8, svi_offload,
                # Block swap
                svi_enable_dit_block_swap, svi_dit_blocks_in_memory,
                svi_enable_text_encoder_block_swap, svi_text_encoder_blocks_in_memory,
                svi_enable_refiner_block_swap, svi_refiner_blocks_in_memory,
                svi_ffn_chunk_size, svi_enable_activation_offload, svi_temporal_chunk_size,
                # LoRA
                svi_lora_folder, svi_lora_dropdown, svi_lora_strength,
                # Output
                svi_disable_audio, svi_output_path, svi_batch_size,
                # Preview Generation
                svi_enable_preview, svi_preview_interval,
            ],
            outputs=[svi_output_gallery, svi_preview_gallery, svi_batch_progress, svi_progress_text]
        )

        # SVI-LTX Stop button
        svi_stop_btn.click(
            fn=stop_generation,
            outputs=[svi_batch_progress]
        )

        # SVI-LTX Random seed button
        svi_random_seed_btn.click(
            fn=lambda: -1,
            outputs=[svi_seed]
        )

        # SVI-LTX LoRA refresh
        svi_lora_refresh_btn.click(
            fn=refresh_lora_dropdown,
            inputs=[svi_lora_folder],
            outputs=[svi_lora_dropdown]
        )

        # SVI-LTX Block swap visibility toggles
        svi_enable_dit_block_swap.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[svi_enable_dit_block_swap],
            outputs=[svi_dit_blocks_in_memory]
        )
        svi_enable_text_encoder_block_swap.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[svi_enable_text_encoder_block_swap],
            outputs=[svi_text_encoder_blocks_in_memory]
        )
        svi_enable_refiner_block_swap.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[svi_enable_refiner_block_swap],
            outputs=[svi_refiner_blocks_in_memory]
        )

        # SVI-LTX Image dimension handlers
        svi_input_image.change(
            fn=update_image_dimensions,
            inputs=[svi_input_image],
            outputs=[svi_original_dims, svi_width, svi_height]
        )

        svi_scale_slider.change(
            fn=update_resolution_from_scale,
            inputs=[svi_scale_slider, svi_original_dims],
            outputs=[svi_width, svi_height]
        )

        svi_calc_width_btn.click(
            fn=calculate_width_from_height,
            inputs=[svi_height, svi_original_dims],
            outputs=[svi_width]
        )

        svi_calc_height_btn.click(
            fn=calculate_height_from_width,
            inputs=[svi_width, svi_original_dims],
            outputs=[svi_height]
        )

        # SVI-LTX Video extension dimension update
        def update_svi_video_dimensions(video_path):
            """Update dimensions when video is uploaded for extension."""
            if not video_path or not os.path.exists(video_path):
                return "", gr.update(), gr.update(), gr.update(), gr.update()
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                # Round to nearest 32
                w = round(w / 32) * 32
                h = round(h / 32) * 32
                fps = max(1, int(round(fps)))
                return f"{w}x{h}", gr.update(value=w), gr.update(value=h), gr.update(), gr.update(value=fps)
            except Exception:
                return "", gr.update(), gr.update(), gr.update(), gr.update()

        svi_extend_video.change(
            fn=update_svi_video_dimensions,
            inputs=[svi_extend_video],
            outputs=[svi_original_dims, svi_width, svi_height, svi_num_frames, svi_frame_rate]
        )

        # =================================================================
        # Save/Load Defaults
        # =================================================================
        lt1_ui_default_components_ORDERED_LIST = [
            # Prompts
            prompt, negative_prompt,
            # Model paths
            checkpoint_path, distilled_checkpoint, stage2_checkpoint, gemma_root,
            spatial_upsampler_path, vae_path, distilled_lora_path, distilled_lora_strength,
            # Generation parameters
            mode, pipeline, sampler, stage2_sampler, width, height, num_frames, frame_rate,
            cfg_guidance_scale, num_inference_steps, stage2_steps, stage2_strength, stage3_strength, seed,
            stg_scale, stg_blocks, stg_mode,
            kandinsky_scheduler, kandinsky_scheduler_scale,
            # Image conditioning (not input_image itself - that's a file upload)
            image_frame_idx, image_strength, image_crf,
            end_image_strength, end_image_crf,
            # Anchor conditioning
            anchor_interval, anchor_strength, anchor_decay, anchor_crf,
            # Refine settings
            refine_strength,
            # Audio and prompt
            disable_audio, audio_strength, v2v_audio_mode, v2v_audio_strength, enhance_prompt,
            # Memory optimization
            offload, enable_fp8,
            enable_dit_block_swap, dit_blocks_in_memory,
            enable_text_encoder_block_swap, text_encoder_blocks_in_memory,
            enable_refiner_block_swap, refiner_blocks_in_memory,
            ffn_chunk_size, enable_activation_offload, temporal_chunk_size,
            # LoRAs
            lora_folder,
            user_lora_1, user_lora_strength_1, user_lora_stage_1,
            user_lora_2, user_lora_strength_2, user_lora_stage_2,
            user_lora_3, user_lora_strength_3, user_lora_stage_3,
            user_lora_4, user_lora_strength_4, user_lora_stage_4,
            # Output
            save_path, batch_size,
            # Scale slider
            scale_slider,
            # V2A Mode
            v2a_mode, v2a_strength,
            # Video Joining (video paths not saved, only parameters)
            v2v_join_frames_check1, v2v_join_frames_check2,
            v2v_join_preserve1, v2v_join_preserve2,
            v2v_join_transition_time,
            v2v_join_steps, v2v_join_terminal,
        ]

        lt1_ui_default_keys = [
            # Prompts
            "prompt", "negative_prompt",
            # Model paths
            "checkpoint_path", "distilled_checkpoint", "stage2_checkpoint", "gemma_root",
            "spatial_upsampler_path", "vae_path", "distilled_lora_path", "distilled_lora_strength",
            # Generation parameters
            "mode", "pipeline", "sampler", "stage2_sampler", "width", "height", "num_frames", "frame_rate",
            "cfg_guidance_scale", "num_inference_steps", "stage2_steps", "stage2_strength", "stage3_strength", "seed",
            "stg_scale", "stg_blocks", "stg_mode",
            "kandinsky_scheduler", "kandinsky_scheduler_scale",
            # Image conditioning
            "image_frame_idx", "image_strength", "image_crf",
            "end_image_strength", "end_image_crf",
            # Anchor conditioning
            "anchor_interval", "anchor_strength", "anchor_decay", "anchor_crf",
            # Refine settings
            "refine_strength",
            # Audio and prompt
            "disable_audio", "audio_strength", "v2v_audio_mode", "v2v_audio_strength", "enhance_prompt",
            # Memory optimization
            "offload", "enable_fp8",
            "enable_dit_block_swap", "dit_blocks_in_memory",
            "enable_text_encoder_block_swap", "text_encoder_blocks_in_memory",
            "enable_refiner_block_swap", "refiner_blocks_in_memory",
            "ffn_chunk_size", "enable_activation_offload", "temporal_chunk_size",
            # LoRAs
            "lora_folder",
            "user_lora_1", "user_lora_strength_1", "user_lora_stage_1",
            "user_lora_2", "user_lora_strength_2", "user_lora_stage_2",
            "user_lora_3", "user_lora_strength_3", "user_lora_stage_3",
            "user_lora_4", "user_lora_strength_4", "user_lora_stage_4",
            # Output
            "save_path", "batch_size",
            # Scale slider
            "scale_slider",
            # V2A Mode
            "v2a_mode", "v2a_strength",
            # Video Joining
            "v2v_join_frames_check1", "v2v_join_frames_check2",
            "v2v_join_preserve1", "v2v_join_preserve2",
            "v2v_join_transition_time",
            "v2v_join_steps", "v2v_join_terminal",
        ]

        def save_lt1_defaults(*values):
            os.makedirs(UI_CONFIGS_DIR, exist_ok=True)
            settings_to_save = {}
            for i, key in enumerate(lt1_ui_default_keys):
                settings_to_save[key] = values[i]
            try:
                with open(LT1_DEFAULTS_FILE, 'w') as f:
                    json.dump(settings_to_save, f, indent=2)
                return "LTX defaults saved successfully."
            except Exception as e:
                return f"Error saving LTX defaults: {e}"

        def load_lt1_defaults(request: gr.Request = None):
            lora_folder_val = "lora"
            lora_choices = get_ltx_lora_options(lora_folder_val)

            if not os.path.exists(LT1_DEFAULTS_FILE):
                if request:
                    return [gr.update()] * len(lt1_ui_default_keys) + ["No defaults file found."]
                else:
                    return [gr.update()] * len(lt1_ui_default_keys) + [""]

            try:
                with open(LT1_DEFAULTS_FILE, 'r') as f:
                    loaded_settings = json.load(f)
            except Exception as e:
                return [gr.update()] * len(lt1_ui_default_keys) + [f"Error loading defaults: {e}"]

            # Update lora folder from settings
            lora_folder_val = loaded_settings.get("lora_folder", "lora")
            lora_choices = get_ltx_lora_options(lora_folder_val)

            updates = []
            for i, key in enumerate(lt1_ui_default_keys):
                component = lt1_ui_default_components_ORDERED_LIST[i]
                default_value_from_component = None
                if hasattr(component, 'value'):
                    default_value_from_component = component.value

                value_to_set = loaded_settings.get(key, default_value_from_component)

                # Special handling for LoRA dropdowns
                if key in ("user_lora_1", "user_lora_2", "user_lora_3", "user_lora_4"):
                    if value_to_set not in lora_choices:
                        value_to_set = "None"
                    updates.append(gr.update(choices=lora_choices, value=value_to_set))
                else:
                    updates.append(gr.update(value=value_to_set))

            return updates + ["LTX defaults loaded successfully."]

        lt1_save_defaults_btn.click(
            fn=save_lt1_defaults,
            inputs=lt1_ui_default_components_ORDERED_LIST,
            outputs=[lt1_defaults_status]
        )
        lt1_load_defaults_btn.click(
            fn=load_lt1_defaults,
            inputs=None,
            outputs=lt1_ui_default_components_ORDERED_LIST + [lt1_defaults_status]
        )

        def initial_load_lt1_defaults():
            results_and_status = load_lt1_defaults(None)
            return results_and_status[:-1]

        demo.load(
            fn=initial_load_lt1_defaults,
            inputs=None,
            outputs=lt1_ui_default_components_ORDERED_LIST
        )

        # =================================================================
        # SVI-LTX Save/Load Defaults
        # =================================================================
        SVI_LTX_DEFAULTS_FILE = os.path.join(UI_CONFIGS_DIR, "svi_ltx_defaults.json")

        svi_ltx_ui_default_components_ORDERED_LIST = [
            # Prompts
            svi_prompt1, svi_prompt2, svi_prompt3, svi_prompt4,
            svi_prompt5, svi_prompt6, svi_prompt7, svi_prompt8,
            svi_negative_prompt,
            # SVI settings
            svi_num_clips, svi_overlap_frames, svi_num_motion_latent,
            svi_num_motion_frame, svi_seed_multiplier,
            # Generation params
            svi_width, svi_height, svi_num_frames, svi_frame_rate,
            svi_inference_steps, svi_cfg_scale, svi_seed,
            # Anchor settings
            svi_anchor_interval, svi_anchor_strength, svi_anchor_decay,
            # Model settings
            svi_checkpoint_path, svi_gemma_root, svi_spatial_upsampler,
            svi_distilled_lora, svi_distilled_lora_strength, svi_one_stage,
            svi_enable_fp8, svi_offload,
            # Block swap
            svi_enable_dit_block_swap, svi_dit_blocks_in_memory,
            svi_enable_text_encoder_block_swap, svi_text_encoder_blocks_in_memory,
            svi_enable_refiner_block_swap, svi_refiner_blocks_in_memory,
            svi_ffn_chunk_size, svi_enable_activation_offload, svi_temporal_chunk_size,
            # LoRA
            svi_lora_folder, svi_lora_dropdown, svi_lora_strength,
            # Output
            svi_disable_audio, svi_output_path, svi_batch_size,
            # Video extension
            svi_frames_to_check, svi_prepend_original,
            # Scale slider
            svi_scale_slider,
        ]

        svi_ltx_ui_default_keys = [
            # Prompts
            "svi_prompt1", "svi_prompt2", "svi_prompt3", "svi_prompt4",
            "svi_prompt5", "svi_prompt6", "svi_prompt7", "svi_prompt8",
            "svi_negative_prompt",
            # SVI settings
            "svi_num_clips", "svi_overlap_frames", "svi_num_motion_latent",
            "svi_num_motion_frame", "svi_seed_multiplier",
            # Generation params
            "svi_width", "svi_height", "svi_num_frames", "svi_frame_rate",
            "svi_inference_steps", "svi_cfg_scale", "svi_seed",
            # Anchor settings
            "svi_anchor_interval", "svi_anchor_strength", "svi_anchor_decay",
            # Model settings
            "svi_checkpoint_path", "svi_gemma_root", "svi_spatial_upsampler",
            "svi_distilled_lora", "svi_distilled_lora_strength", "svi_one_stage",
            "svi_enable_fp8", "svi_offload",
            # Block swap
            "svi_enable_dit_block_swap", "svi_dit_blocks_in_memory",
            "svi_enable_text_encoder_block_swap", "svi_text_encoder_blocks_in_memory",
            "svi_enable_refiner_block_swap", "svi_refiner_blocks_in_memory",
            "svi_ffn_chunk_size", "svi_enable_activation_offload", "svi_temporal_chunk_size",
            # LoRA
            "svi_lora_folder", "svi_lora_dropdown", "svi_lora_strength",
            # Output
            "svi_disable_audio", "svi_output_path", "svi_batch_size",
            # Video extension
            "svi_frames_to_check", "svi_prepend_original",
            # Scale slider
            "svi_scale_slider",
        ]

        def save_svi_ltx_defaults(*values):
            os.makedirs(UI_CONFIGS_DIR, exist_ok=True)
            settings_to_save = {}
            for i, key in enumerate(svi_ltx_ui_default_keys):
                settings_to_save[key] = values[i]
            try:
                with open(SVI_LTX_DEFAULTS_FILE, 'w') as f:
                    json.dump(settings_to_save, f, indent=2)
                return "SVI-LTX defaults saved successfully."
            except Exception as e:
                return f"Error saving SVI-LTX defaults: {e}"

        def load_svi_ltx_defaults(request: gr.Request = None):
            lora_folder_val = "lora"
            lora_choices = get_ltx_lora_options(lora_folder_val)

            if not os.path.exists(SVI_LTX_DEFAULTS_FILE):
                if request:
                    return [gr.update()] * len(svi_ltx_ui_default_keys) + ["No SVI-LTX defaults file found."]
                else:
                    return [gr.update()] * len(svi_ltx_ui_default_keys) + [""]

            try:
                with open(SVI_LTX_DEFAULTS_FILE, 'r') as f:
                    loaded_settings = json.load(f)
            except Exception as e:
                return [gr.update()] * len(svi_ltx_ui_default_keys) + [f"Error loading SVI-LTX defaults: {e}"]

            # Update lora folder from settings
            lora_folder_val = loaded_settings.get("svi_lora_folder", "lora")
            lora_choices = get_ltx_lora_options(lora_folder_val)

            updates = []
            for i, key in enumerate(svi_ltx_ui_default_keys):
                component = svi_ltx_ui_default_components_ORDERED_LIST[i]
                default_value_from_component = None
                if hasattr(component, 'value'):
                    default_value_from_component = component.value

                value_to_set = loaded_settings.get(key, default_value_from_component)

                # Special handling for LoRA dropdown
                if key == "svi_lora_dropdown":
                    if value_to_set not in lora_choices:
                        value_to_set = "None"
                    updates.append(gr.update(choices=lora_choices, value=value_to_set))
                else:
                    updates.append(gr.update(value=value_to_set))

            return updates + ["SVI-LTX defaults loaded successfully."]

        svi_save_defaults_btn.click(
            fn=save_svi_ltx_defaults,
            inputs=svi_ltx_ui_default_components_ORDERED_LIST,
            outputs=[svi_batch_progress]
        )
        svi_load_defaults_btn.click(
            fn=load_svi_ltx_defaults,
            inputs=None,
            outputs=svi_ltx_ui_default_components_ORDERED_LIST + [svi_batch_progress]
        )

        def initial_load_svi_ltx_defaults():
            results_and_status = load_svi_ltx_defaults(None)
            return results_and_status[:-1]

        demo.load(
            fn=initial_load_svi_ltx_defaults,
            inputs=None,
            outputs=svi_ltx_ui_default_components_ORDERED_LIST
        )

        # =================================================================
        # Extension Tab Save/Load Defaults
        # =================================================================
        EXT_DEFAULTS_FILE = os.path.join(UI_CONFIGS_DIR, "ext_defaults.json")

        ext_ui_default_components_ORDERED_LIST = [
            # Extension settings
            ext_extend_seconds, ext_preserve_seconds, ext_steps, ext_cfg,
            ext_preserve_strength, ext_skip_stage2, ext_batch_size,
            # Model paths
            ext_checkpoint_path, ext_distilled_checkpoint,
            ext_gemma_root, ext_spatial_upsampler_path, ext_vae_path,
            ext_distilled_lora_path, ext_distilled_lora_strength,
            # Memory optimization
            ext_offload, ext_enable_fp8,
            ext_enable_dit_block_swap, ext_dit_blocks_in_memory,
            ext_enable_text_encoder_block_swap, ext_text_encoder_blocks_in_memory,
            ext_enable_refiner_block_swap, ext_refiner_blocks_in_memory,
            ext_enable_activation_offload,
            # LoRA
            ext_lora_folder,
            ext_user_lora_1, ext_user_lora_strength_1, ext_user_lora_stage_1,
            ext_user_lora_2, ext_user_lora_strength_2, ext_user_lora_stage_2,
            ext_user_lora_3, ext_user_lora_strength_3, ext_user_lora_stage_3,
            ext_user_lora_4, ext_user_lora_strength_4, ext_user_lora_stage_4,
            # Output
            ext_save_path,
        ]

        ext_ui_default_keys = [
            "ext_extend_seconds", "ext_preserve_seconds", "ext_steps", "ext_cfg",
            "ext_preserve_strength", "ext_skip_stage2", "ext_batch_size",
            "ext_checkpoint_path", "ext_distilled_checkpoint",
            "ext_gemma_root", "ext_spatial_upsampler_path", "ext_vae_path",
            "ext_distilled_lora_path", "ext_distilled_lora_strength",
            "ext_offload", "ext_enable_fp8",
            "ext_enable_dit_block_swap", "ext_dit_blocks_in_memory",
            "ext_enable_text_encoder_block_swap", "ext_text_encoder_blocks_in_memory",
            "ext_enable_refiner_block_swap", "ext_refiner_blocks_in_memory",
            "ext_enable_activation_offload",
            "ext_lora_folder",
            "ext_user_lora_1", "ext_user_lora_strength_1", "ext_user_lora_stage_1",
            "ext_user_lora_2", "ext_user_lora_strength_2", "ext_user_lora_stage_2",
            "ext_user_lora_3", "ext_user_lora_strength_3", "ext_user_lora_stage_3",
            "ext_user_lora_4", "ext_user_lora_strength_4", "ext_user_lora_stage_4",
            "ext_save_path",
        ]

        def save_ext_defaults(*values):
            os.makedirs(UI_CONFIGS_DIR, exist_ok=True)
            settings_to_save = {}
            for i, key in enumerate(ext_ui_default_keys):
                settings_to_save[key] = values[i]
            try:
                with open(EXT_DEFAULTS_FILE, 'w') as f:
                    json.dump(settings_to_save, f, indent=2)
                return "Extension defaults saved successfully."
            except Exception as e:
                return f"Error saving Extension defaults: {e}"

        def load_ext_defaults(request: gr.Request = None):
            lora_folder_val = "lora"
            lora_choices = get_ltx_lora_options(lora_folder_val)

            if not os.path.exists(EXT_DEFAULTS_FILE):
                if request:
                    return [gr.update()] * len(ext_ui_default_keys) + ["No defaults file found."]
                else:
                    return [gr.update()] * len(ext_ui_default_keys) + [""]

            try:
                with open(EXT_DEFAULTS_FILE, 'r') as f:
                    loaded_settings = json.load(f)
            except Exception as e:
                return [gr.update()] * len(ext_ui_default_keys) + [f"Error loading defaults: {e}"]

            # Update lora folder from settings
            lora_folder_val = loaded_settings.get("ext_lora_folder", "lora")
            lora_choices = get_ltx_lora_options(lora_folder_val)

            updates = []
            for i, key in enumerate(ext_ui_default_keys):
                component = ext_ui_default_components_ORDERED_LIST[i]
                default_value_from_component = None
                if hasattr(component, 'value'):
                    default_value_from_component = component.value

                value_to_set = loaded_settings.get(key, default_value_from_component)

                # Special handling for LoRA dropdowns
                if key in ("ext_user_lora_1", "ext_user_lora_2", "ext_user_lora_3", "ext_user_lora_4"):
                    if value_to_set not in lora_choices:
                        value_to_set = "None"
                    updates.append(gr.update(choices=lora_choices, value=value_to_set))
                else:
                    updates.append(gr.update(value=value_to_set))

            return updates + ["Extension defaults loaded successfully."]

        ext_save_defaults_btn.click(
            fn=save_ext_defaults,
            inputs=ext_ui_default_components_ORDERED_LIST,
            outputs=[ext_defaults_status]
        )
        ext_load_defaults_btn.click(
            fn=load_ext_defaults,
            inputs=None,
            outputs=ext_ui_default_components_ORDERED_LIST + [ext_defaults_status]
        )

        def initial_load_ext_defaults():
            results_and_status = load_ext_defaults(None)
            return results_and_status[:-1]

        demo.load(
            fn=initial_load_ext_defaults,
            inputs=None,
            outputs=ext_ui_default_components_ORDERED_LIST
        )

        # =================================================================
        # Depth Map Tab Event Handlers
        # =================================================================

        # Toggle input visibility based on input type
        def update_depth_input_visibility(input_type):
            return (
                gr.update(visible=(input_type == "Image")),
                gr.update(visible=(input_type == "Video")),
            )

        depth_input_type.change(
            fn=update_depth_input_visibility,
            inputs=[depth_input_type],
            outputs=[depth_input_image, depth_input_video]
        )

        # Toggle output options visibility based on input/output type
        def update_depth_output_visibility(input_type, output_type):
            # Show num_frames/fps only when converting image to video
            show_video_options = (input_type == "Image" and output_type == "Video")
            # Show max_frames only when processing video
            show_max_frames = (input_type == "Video")
            # Show video output preview when output is video
            show_video_output = (output_type == "Video" or (input_type == "Video" and output_type != "Image"))
            show_image_output = not show_video_output

            return (
                gr.update(visible=show_video_options),  # num_frames
                gr.update(visible=show_video_options),  # fps
                gr.update(visible=show_max_frames),     # max_frames
                gr.update(visible=show_image_output),   # image output
                gr.update(visible=show_video_output),   # video output
            )

        depth_input_type.change(
            fn=lambda i, o: update_depth_output_visibility(i, o),
            inputs=[depth_input_type, depth_output_type],
            outputs=[depth_num_frames, depth_fps, depth_max_frames, depth_output_image, depth_output_video]
        )
        depth_output_type.change(
            fn=lambda i, o: update_depth_output_visibility(i, o),
            inputs=[depth_input_type, depth_output_type],
            outputs=[depth_num_frames, depth_fps, depth_max_frames, depth_output_image, depth_output_video]
        )

        # Generate depth map
        depth_generate_btn.click(
            fn=generate_depth_map,
            inputs=[
                depth_input_type,
                depth_input_image,
                depth_input_video,
                depth_output_type,
                depth_colorize,
                depth_num_frames,
                depth_fps,
                depth_max_frames,
                depth_width,
                depth_height,
            ],
            outputs=[depth_output_image, depth_output_video, depth_status]
        )

        # Store depth map output path for sending to generation tab
        depth_output_path = gr.State(value=None)

        # Update stored path when output changes
        def store_depth_path(img_path, vid_path):
            return img_path if img_path else vid_path

        depth_output_image.change(
            fn=lambda x: x,
            inputs=[depth_output_image],
            outputs=[depth_output_path]
        )
        depth_output_video.change(
            fn=lambda x: x,
            inputs=[depth_output_video],
            outputs=[depth_output_path]
        )

        # Send depth map to generation tab
        def send_depth_to_generation(img_path, vid_path):
            """Send generated depth map to the Generation tab's depth control section."""
            if vid_path:
                status_info = update_depth_control_status(vid_path, None)
                return (
                    gr.Tabs(selected="gen_tab"),
                    gr.update(value=vid_path),  # depth_control_video
                    gr.update(value=None),      # depth_control_image
                    "Depth video sent to Generation tab",
                    status_info
                )
            elif img_path:
                status_info = update_depth_control_status(None, img_path)
                return (
                    gr.Tabs(selected="gen_tab"),
                    gr.update(value=None),      # depth_control_video
                    gr.update(value=img_path),  # depth_control_image
                    "Depth image sent to Generation tab",
                    status_info
                )
            else:
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    "No depth map to send - generate one first",
                    gr.update()
                )

        depth_send_to_gen_btn.click(
            fn=send_depth_to_generation,
            inputs=[depth_output_image, depth_output_video],
            outputs=[tabs, depth_control_video, depth_control_image, depth_status, depth_control_status]
        )

        # Depth control status updates
        depth_control_video.change(
            fn=update_depth_control_status,
            inputs=[depth_control_video, depth_control_image],
            outputs=[depth_control_status]
        )
        depth_control_image.change(
            fn=update_depth_control_status,
            inputs=[depth_control_video, depth_control_image],
            outputs=[depth_control_status]
        )

        # =================================================================
        # Frame Interpolation Event Handlers
        # =================================================================
        interp_generate_btn.click(
            fn=interpolate_video,
            inputs=[
                interp_input_video,
                interp_model_variant,
                interp_checkpoint_path,
                interp_config_path,
                interp_factor,
                interp_ds_scale,
                interp_output_fps,
                interp_raft_iters,
                interp_pyr_level,
                interp_seed,
            ],
            outputs=[interp_output_video, interp_status, interp_progress]
        )

        # Upscaling event handler
        upscale_btn.click(
            fn=upscale_video,
            inputs=[
                interp_input_video,  # Use same input video
                upscale_model,
                upscale_model_path,
                upscale_tile_size,
                upscale_half,
                motion_blur_enable,
                motion_blur_strength,
                motion_blur_samples,
                upscale_crf,
                interp_seed,  # Reuse same seed
            ],
            outputs=[interp_output_video, interp_status, interp_progress]
        )

        # =================================================================
        # Extension Tab Event Handlers
        # =================================================================

        # Random seed button
        ext_random_seed_btn.click(
            fn=lambda: -1,
            outputs=[ext_seed]
        )

        # LoRA refresh (updates all 4 dropdowns)
        def ext_refresh_all_lora_dropdowns(folder):
            choices = get_ltx_lora_options(folder)
            return [gr.update(choices=choices) for _ in range(4)]

        ext_lora_refresh_btn.click(
            fn=ext_refresh_all_lora_dropdowns,
            inputs=[ext_lora_folder],
            outputs=[ext_user_lora_1, ext_user_lora_2, ext_user_lora_3, ext_user_lora_4]
        )

        # Block swap visibility toggles
        ext_enable_dit_block_swap.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[ext_enable_dit_block_swap],
            outputs=[ext_dit_blocks_in_memory]
        )
        ext_enable_text_encoder_block_swap.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[ext_enable_text_encoder_block_swap],
            outputs=[ext_text_encoder_blocks_in_memory]
        )
        ext_enable_refiner_block_swap.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[ext_enable_refiner_block_swap],
            outputs=[ext_refiner_blocks_in_memory]
        )

        # Stop button
        ext_stop_btn.click(
            fn=stop_generation,
            outputs=[ext_status_text]
        )

        # Generate extension button
        ext_generate_btn.click(
            fn=generate_extension_video,
            inputs=[
                # Extension-specific parameters
                ext_input_video, ext_prompt, ext_negative_prompt,
                ext_extend_seconds, ext_preserve_seconds, ext_seed, ext_steps, ext_cfg,
                ext_preserve_strength, ext_skip_stage2,
                # Model paths
                ext_checkpoint_path, ext_distilled_checkpoint,
                ext_gemma_root, ext_spatial_upsampler_path, ext_vae_path,
                ext_distilled_lora_path, ext_distilled_lora_strength,
                # Memory optimization
                ext_offload, ext_enable_fp8,
                ext_enable_dit_block_swap, ext_dit_blocks_in_memory,
                ext_enable_text_encoder_block_swap, ext_text_encoder_blocks_in_memory,
                ext_enable_refiner_block_swap, ext_refiner_blocks_in_memory,
                ext_enable_activation_offload,
                # LoRA
                ext_lora_folder,
                ext_user_lora_1, ext_user_lora_strength_1, ext_user_lora_stage_1,
                ext_user_lora_2, ext_user_lora_strength_2, ext_user_lora_stage_2,
                ext_user_lora_3, ext_user_lora_strength_3, ext_user_lora_stage_3,
                ext_user_lora_4, ext_user_lora_strength_4, ext_user_lora_stage_4,
                # Output
                ext_save_path,
                # Batching
                ext_batch_size,
            ],
            outputs=[ext_output_gallery, ext_status_text, ext_progress_text]
        )

        return demo


# =============================================================================
# Port Detection
# =============================================================================

def find_available_port(start_port: int = 7860, max_attempts: int = 100) -> int:
    """Find the first available port starting from start_port."""
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find available port in range {start_port}-{start_port + max_attempts}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LTX-2 Video Generation UI")
    parser.add_argument("--port", type=int, default=None,
                        help="Server port. Auto-detects available port if not specified.")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link.")
    args = parser.parse_args()

    # Determine port (auto-detect if not specified)
    if args.port is not None:
        port = args.port
    else:
        port = find_available_port(7860)

    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    print(f"[lt1.py] Starting on port {port}, CUDA_VISIBLE_DEVICES={cuda_devices}")

    demo = create_interface()
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=args.share,
        inbrowser=False
    )
