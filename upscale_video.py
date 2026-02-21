#!/usr/bin/env python3
"""
Standalone video upscaling script with motion blur synthesis.
Runs in a subprocess to ensure complete VRAM cleanup on exit.

Supports: Real-ESRGAN, SwinIR (via Spandrel), BasicVSR++
Motion blur: Optical flow based using RAFT from GIMM-VFI
"""

import argparse
import os
import sys
import tempfile
import shutil
import time
from pathlib import Path
import subprocess


def extract_video_frames(video_path: str, output_dir: str):
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


def frames_to_video(frame_dir: str, output_path: str, fps: float, audio_source: str = None, crf: int = 18):
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
            "-crf", str(crf),
            temp_video
        ]
        subprocess.run(cmd, capture_output=True)

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
            tempo = orig_duration / new_duration
            tempo = max(0.5, min(2.0, tempo))

            cmd = [
                "ffmpeg", "-y",
                "-i", temp_video,
                "-i", audio_source,
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-c:v", "copy",
                "-c:a", "aac",
                "-filter:a", f"atempo={tempo}",
                "-shortest",
                output_path
            ]
            subprocess.run(cmd, capture_output=True)
            os.remove(temp_video)
        else:
            shutil.move(temp_video, output_path)
    else:
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(frame_dir, "%05d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", str(crf),
            output_path
        ]
        subprocess.run(cmd, capture_output=True)


# Backwarp implementation (from GIMM-VFI/bim_vfi/backwarp.py)
_backwarp_cache = {}

def backwarp(tenIn, tenFlow, mode='bilinear'):
    """Warp input tensor by flow field."""
    import torch
    cache_key = f'grid_{tenFlow.dtype}_{tenFlow.device}_{tenFlow.shape[2]}_{tenFlow.shape[3]}'
    if cache_key not in _backwarp_cache:
        tenHor = torch.linspace(-1.0, 1.0, tenFlow.shape[3], dtype=tenFlow.dtype, device=tenFlow.device)
        tenHor = tenHor.view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(-1.0, 1.0, tenFlow.shape[2], dtype=tenFlow.dtype, device=tenFlow.device)
        tenVer = tenVer.view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])
        _backwarp_cache[cache_key] = torch.cat([tenHor, tenVer], 1)

    if tenFlow.shape[3] == tenFlow.shape[2]:
        tenFlow = tenFlow * (2.0 / (tenFlow.shape[3] - 1.0))
    else:
        scale = torch.tensor([2.0 / (tenFlow.shape[3] - 1.0), 2.0 / (tenFlow.shape[2] - 1.0)],
                           dtype=tenFlow.dtype, device=tenFlow.device).view(1, 2, 1, 1)
        tenFlow = tenFlow * scale

    grid = (_backwarp_cache[cache_key] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(tenIn, grid, mode=mode, padding_mode='zeros', align_corners=True)


def apply_motion_blur_flow(frame, flow, strength=1.0, samples=7):
    """
    Apply motion blur based on optical flow vectors.

    Args:
        frame: [B, C, H, W] input frame tensor
        flow: [B, 2, H, W] optical flow (dx, dy per pixel)
        strength: blur length multiplier
        samples: number of samples along blur line (odd number recommended)

    Returns:
        Motion-blurred frame tensor
    """
    import torch

    # Gaussian weights for smooth blur falloff
    t_values = torch.linspace(-2, 2, samples, device=frame.device)
    weights = torch.exp(-t_values ** 2)
    weights = weights / weights.sum()

    blurred = torch.zeros_like(frame)

    for i, t in enumerate(torch.linspace(-0.5, 0.5, samples, device=frame.device)):
        offset_flow = flow * t * strength
        warped = backwarp(frame, offset_flow)
        blurred = blurred + warped * weights[i]

    return blurred


def load_raft_model(device):
    """Load RAFT from GIMM-VFI for optical flow estimation."""
    import torch

    script_dir = os.path.dirname(os.path.abspath(__file__))
    gimm_dir = os.path.join(script_dir, "GIMM-VFI")
    gimm_src = os.path.join(gimm_dir, "src")

    if gimm_src not in sys.path:
        sys.path.insert(0, gimm_src)

    from models.generalizable_INR.raft.raft import RAFT

    # Create args for RAFT
    class RAFTArgs:
        small = False
        mixed_precision = False
        alternate_corr = False

    model = RAFT(RAFTArgs())
    ckpt_path = os.path.join(gimm_dir, "pretrained_ckpt/raft-things.pth")

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # Handle different checkpoint formats
        if isinstance(ckpt, dict):
            state_dict = ckpt.get("state_dict", ckpt)
        else:
            state_dict = ckpt
        # Remove 'module.' prefix if present
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"WARNING: RAFT checkpoint not found at {ckpt_path}")

    model = model.to(device)
    model.eval()
    return model


def estimate_flow(raft_model, frame1, frame2, iters=20):
    """Estimate optical flow between two frames using RAFT."""
    import torch

    with torch.no_grad():
        # RAFT expects [B, C, H, W] in range [0, 255]
        frame1_input = frame1 * 255.0
        frame2_input = frame2 * 255.0

        # Run RAFT
        _, flow = raft_model(frame1_input, frame2_input, iters=iters, test_mode=True)

    return flow


def load_spandrel_model(model_path, device, half=False):
    """Load ESRGAN/SwinIR model via Spandrel universal loader."""
    import torch
    try:
        from spandrel import ModelLoader, ImageModelDescriptor
    except ImportError:
        print("ERROR: spandrel not installed. Run: pip install spandrel")
        sys.exit(1)

    print(f"PROGRESS: Loading model via Spandrel...")
    loader = ModelLoader()
    model_descriptor = loader.load_from_file(model_path)

    if not isinstance(model_descriptor, ImageModelDescriptor):
        print("ERROR: Model is not an image upscaling model")
        sys.exit(1)

    model = model_descriptor.model.eval()
    if half:
        model = model.half()
    model = model.to(device)

    scale = model_descriptor.scale
    return model, scale


def upscale_frame_tiled(model, frame, tile_size=512, tile_pad=32, scale=2):
    """Upscale a frame with optional tiling for memory efficiency."""
    import torch

    if tile_size == 0:
        # No tiling
        with torch.no_grad():
            return model(frame)

    _, _, h, w = frame.shape

    # If frame is smaller than tile, just process it directly
    if h <= tile_size and w <= tile_size:
        with torch.no_grad():
            return model(frame)

    # Tiled processing
    output_h, output_w = h * scale, w * scale
    output = torch.zeros((1, 3, output_h, output_w), dtype=frame.dtype, device=frame.device)
    weight = torch.zeros((1, 1, output_h, output_w), dtype=frame.dtype, device=frame.device)

    for y in range(0, h, tile_size - tile_pad * 2):
        for x in range(0, w, tile_size - tile_pad * 2):
            # Extract tile with padding
            y_start = max(0, y - tile_pad)
            x_start = max(0, x - tile_pad)
            y_end = min(h, y + tile_size - tile_pad)
            x_end = min(w, x + tile_size - tile_pad)

            tile = frame[:, :, y_start:y_end, x_start:x_end]

            with torch.no_grad():
                tile_output = model(tile)

            # Calculate output positions
            out_y_start = y_start * scale
            out_x_start = x_start * scale
            out_y_end = y_end * scale
            out_x_end = x_end * scale

            # Remove padding from output
            pad_y = (y - y_start) * scale
            pad_x = (x - x_start) * scale
            tile_h = (y_end - y_start) * scale
            tile_w = (x_end - x_start) * scale

            output[:, :, out_y_start:out_y_end, out_x_start:out_x_end] += tile_output
            weight[:, :, out_y_start:out_y_end, out_x_start:out_x_end] += 1

    output = output / weight.clamp(min=1)
    return output


def upscale_spandrel(args):
    """Frame-by-frame upscaling with ESRGAN/SwinIR via Spandrel."""
    import torch
    import numpy as np
    import cv2
    from PIL import Image

    print(f"PROGRESS: Starting upscaling with {args.model_type}...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.half else torch.float32

    # Load upscaler model
    model, scale = load_spandrel_model(args.model_path, device, args.half)
    print(f"PROGRESS: Model loaded, scale factor: {scale}x")

    # Load RAFT if motion blur enabled
    raft_model = None
    if args.motion_blur:
        print(f"PROGRESS: Loading RAFT for motion blur...")
        raft_model = load_raft_model(device)

    # Create temp directories
    temp_dir = tempfile.mkdtemp(prefix="upscale_")
    input_frames_dir = os.path.join(temp_dir, "input_frames")
    output_frames_dir = os.path.join(temp_dir, "output_frames")
    os.makedirs(input_frames_dir, exist_ok=True)
    os.makedirs(output_frames_dir, exist_ok=True)

    try:
        # Extract frames
        print(f"PROGRESS: Extracting frames...")
        frame_paths, fps = extract_video_frames(args.input, input_frames_dir)
        print(f"PROGRESS: Extracted {len(frame_paths)} frames at {fps:.2f} FPS")

        if len(frame_paths) < 1:
            print("ERROR: Video must have at least 1 frame")
            return 1

        def load_frame(path):
            img = Image.open(path).convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            return tensor.to(device, dtype)

        def save_frame(tensor, path):
            arr = tensor[0].cpu().float().numpy().transpose(1, 2, 0)
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, arr_bgr)

        # Process frames
        total_frames = len(frame_paths)
        prev_upscaled = None

        for i, frame_path in enumerate(frame_paths):
            progress_pct = int((i / total_frames) * 100)
            print(f"PROGRESS: Upscaling frame {i + 1}/{total_frames} ({progress_pct}%)")

            # Load and upscale frame
            frame = load_frame(frame_path)
            upscaled = upscale_frame_tiled(model, frame, args.tile_size, scale=scale)

            # Apply motion blur if enabled
            if args.motion_blur and prev_upscaled is not None and raft_model is not None:
                # Estimate flow between previous and current upscaled frames
                # Convert to float32 for RAFT
                prev_float = prev_upscaled.float()
                curr_float = upscaled.float()
                flow = estimate_flow(raft_model, prev_float, curr_float, iters=12)

                # Apply motion blur to current frame
                upscaled_float = upscaled.float()
                upscaled_blurred = apply_motion_blur_flow(
                    upscaled_float, flow,
                    strength=args.blur_strength,
                    samples=args.blur_samples
                )
                upscaled = upscaled_blurred.to(dtype)

            # Save upscaled frame
            output_path = os.path.join(output_frames_dir, f"{i:05d}.png")
            save_frame(upscaled, output_path)

            # Store for next iteration's motion blur
            prev_upscaled = upscaled.clone()

            # Clear cache periodically
            if i % 10 == 0:
                torch.cuda.empty_cache()

        print(f"PROGRESS: Encoding output video...")

        # Reassemble video
        frames_to_video(output_frames_dir, args.output, fps, audio_source=args.input, crf=args.crf)

        print(f"PROGRESS: Done! Upscaled {total_frames} frames at {scale}x")
        print(f"OUTPUT: {args.output}")
        return 0

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def upscale_basicvsr(args):
    """Temporal video upscaling with BasicVSR++."""
    import torch
    import numpy as np
    import cv2
    from PIL import Image
    from basicvsr_pp import BasicVSRPlusPlus, load_basicvsr_checkpoint

    print("PROGRESS: Starting BasicVSR++ temporal upscaling...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.half else torch.float32

    # Load model
    model = BasicVSRPlusPlus(mid_channels=64, num_blocks=7, is_low_res_input=True)
    load_basicvsr_checkpoint(model, args.model_path)
    if args.half:
        model = model.half()
    model = model.to(device)
    model.eval()
    print("PROGRESS: BasicVSR++ model loaded (4x upscaling)")

    # Load RAFT if motion blur enabled
    raft_model = None
    if args.motion_blur:
        print("PROGRESS: Loading RAFT for motion blur...")
        raft_model = load_raft_model(device)

    # Create temp directories
    temp_dir = tempfile.mkdtemp(prefix="upscale_bvsr_")
    input_frames_dir = os.path.join(temp_dir, "input_frames")
    output_frames_dir = os.path.join(temp_dir, "output_frames")
    os.makedirs(input_frames_dir, exist_ok=True)
    os.makedirs(output_frames_dir, exist_ok=True)

    try:
        # Extract frames
        print("PROGRESS: Extracting frames...")
        frame_paths, fps = extract_video_frames(args.input, input_frames_dir)
        total_frames = len(frame_paths)
        print(f"PROGRESS: Extracted {total_frames} frames at {fps:.2f} FPS")

        if total_frames < 2:
            print("ERROR: BasicVSR++ requires at least 2 frames")
            return 1

        # Load all frames as tensor [1, T, 3, H, W]
        print("PROGRESS: Loading frames into tensor...")
        frames = []
        for p in frame_paths:
            img = Image.open(p).convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1)
            frames.append(t)

        lqs = torch.stack(frames, dim=0).unsqueeze(0)  # [1, T, 3, H, W]
        lqs = lqs.to(device, dtype)

        # Process with optional temporal chunking
        chunk_size = args.temporal_chunk
        if chunk_size > 0 and total_frames > chunk_size:
            overlap = min(5, chunk_size // 4)
            all_outputs = []
            start = 0
            while start < total_frames:
                end = min(start + chunk_size, total_frames)
                chunk_start = max(0, start - overlap)
                chunk_end = min(total_frames, end + overlap)

                chunk = lqs[:, chunk_start:chunk_end]

                progress_pct = int((start / total_frames) * 100)
                print(f"PROGRESS: Processing frames {start + 1}-{end}/{total_frames} ({progress_pct}%)")

                with torch.no_grad():
                    output = model(chunk)

                # Keep only the non-overlap portion
                out_start = start - chunk_start
                out_end = out_start + (end - start)
                all_outputs.append(output[:, out_start:out_end].cpu())

                start = end
                torch.cuda.empty_cache()

            output = torch.cat(all_outputs, dim=1)
        else:
            print(f"PROGRESS: Processing all {total_frames} frames through BasicVSR++...")
            with torch.no_grad():
                output = model(lqs)
            output = output.cpu()

        # Free model VRAM
        del model, lqs
        torch.cuda.empty_cache()

        # Save frames (with optional motion blur)
        print("PROGRESS: Saving upscaled frames...")
        prev_upscaled = None
        for i in range(output.shape[1]):
            progress_pct = int((i / output.shape[1]) * 100)
            if i % 10 == 0:
                print(f"PROGRESS: Saving frame {i + 1}/{output.shape[1]} ({progress_pct}%)")

            upscaled = output[:, i].to(device)

            # Apply motion blur if enabled
            if args.motion_blur and prev_upscaled is not None and raft_model is not None:
                prev_float = prev_upscaled.float()
                curr_float = upscaled.float()
                flow = estimate_flow(raft_model, prev_float, curr_float, iters=12)
                upscaled_blurred = apply_motion_blur_flow(
                    curr_float, flow,
                    strength=args.blur_strength,
                    samples=args.blur_samples,
                )
                upscaled = upscaled_blurred.to(dtype)

            prev_upscaled = upscaled.clone()

            frame = upscaled[0].cpu().float().numpy().transpose(1, 2, 0)
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_frames_dir, f"{i:05d}.png"), frame_bgr)

        print("PROGRESS: Encoding output video...")
        frames_to_video(output_frames_dir, args.output, fps,
                        audio_source=args.input, crf=args.crf)

        print(f"PROGRESS: Done! Upscaled {total_frames} frames at 4x with BasicVSR++")
        print(f"OUTPUT: {args.output}")
        return 0

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Video Upscaling with Motion Blur")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--model-type", choices=["esrgan", "swinir", "basicvsr"], default="esrgan",
                        help="Model type")
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint")
    parser.add_argument("--scale", type=int, default=2, help="Scale factor (for reference)")
    parser.add_argument("--tile-size", type=int, default=512,
                        help="Tile size for processing (0=no tiling)")
    parser.add_argument("--half", action="store_true", help="Use FP16 precision")
    parser.add_argument("--crf", type=int, default=18, help="Output video CRF (lower=better quality)")

    # Motion blur options
    parser.add_argument("--motion-blur", action="store_true",
                        help="Enable motion blur to mask deformation artifacts")
    parser.add_argument("--blur-strength", type=float, default=1.0,
                        help="Motion blur strength (0.5-2.0)")
    parser.add_argument("--blur-samples", type=int, default=7,
                        help="Motion blur samples (odd number, 3-15)")

    # BasicVSR++ temporal chunking
    parser.add_argument("--temporal-chunk", type=int, default=0,
                        help="Temporal chunk size for BasicVSR++ (0=process all frames at once)")

    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    # Set seed
    import random
    random.seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.model_type in ["esrgan", "swinir"]:
        return upscale_spandrel(args)
    elif args.model_type == "basicvsr":
        return upscale_basicvsr(args)
    else:
        print(f"ERROR: Unknown model type: {args.model_type}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
