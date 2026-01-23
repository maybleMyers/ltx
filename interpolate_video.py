#!/usr/bin/env python3
"""
Standalone video frame interpolation script.
Runs in a subprocess to ensure complete VRAM cleanup on exit.
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


def frames_to_video(frame_dir: str, output_path: str, fps: float, audio_source: str = None):
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
            "-crf", "18",
            output_path
        ]
        subprocess.run(cmd, capture_output=True)


def interpolate_bim(args):
    """Run BiM-VFI interpolation."""
    import torch
    import numpy as np
    import cv2
    from PIL import Image

    # Add BiM-VFI to path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gimm_dir = os.path.join(script_dir, "GIMM-VFI")
    if gimm_dir not in sys.path:
        sys.path.insert(0, gimm_dir)

    from bim_vfi import BiMVFI

    print(f"PROGRESS: Loading BiM-VFI model...")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiMVFI(pyr_level=3, feat_channels=32)
    model = model.to(device)

    # Load checkpoint
    ckpt_path = args.checkpoint if args.checkpoint else os.path.join(script_dir, "GIMM-VFI/pretrained_ckpt/bim_vfi.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print(f"PROGRESS: Model loaded, extracting frames...")

    # Create temp directories
    temp_dir = tempfile.mkdtemp(prefix="bim_interp_")
    input_frames_dir = os.path.join(temp_dir, "input_frames")
    output_frames_dir = os.path.join(temp_dir, "output_frames")
    os.makedirs(input_frames_dir, exist_ok=True)
    os.makedirs(output_frames_dir, exist_ok=True)

    try:
        # Extract frames
        frame_paths, original_fps = extract_video_frames(args.input, input_frames_dir)
        print(f"PROGRESS: Extracted {len(frame_paths)} frames at {original_fps:.2f} FPS")

        if len(frame_paths) < 2:
            print("ERROR: Video must have at least 2 frames")
            return 1

        # Auto-detect pyr_level
        first_img = Image.open(frame_paths[0])
        width, height = first_img.size
        max_dim = max(width, height)

        if args.pyr_level <= 0:
            if max_dim >= 3840:
                pyr_level = 7
            elif max_dim >= 1920:
                pyr_level = 6
            else:
                pyr_level = 5
            print(f"PROGRESS: Auto-detected pyr_level={pyr_level} for {width}x{height}")
        else:
            pyr_level = args.pyr_level

        # Process frame pairs
        N = args.factor
        total_pairs = len(frame_paths) - 1
        output_frame_idx = 0

        def load_image(img_path):
            img = Image.open(img_path)
            raw_img = np.array(img.convert("RGB"))
            img_tensor = torch.from_numpy(raw_img.copy()).permute(2, 0, 1) / 255.0
            return img_tensor.to(torch.float).unsqueeze(0)

        for pair_idx in range(total_pairs):
            progress_pct = int((pair_idx / total_pairs) * 100)
            print(f"PROGRESS: Interpolating pair {pair_idx + 1}/{total_pairs} ({progress_pct}%)")

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
                    results = model(img0=I0, img1=I1, time_step=time_step, pyr_level=pyr_level)
                    imgt_pred = results["imgt_pred"]

                    frame_np = (imgt_pred[0].cpu().numpy().transpose(1, 2, 0) * 255.0)
                    frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
                    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(output_frames_dir, f"{output_frame_idx:05d}.png"), frame_bgr)
                    output_frame_idx += 1

            # Save second frame
            frame_np = (I1[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_frames_dir, f"{output_frame_idx:05d}.png"), frame_bgr)
            output_frame_idx += 1

            if pair_idx % 10 == 0:
                torch.cuda.empty_cache()

        print(f"PROGRESS: Encoding output video...")

        # Calculate output FPS
        output_fps = args.output_fps if args.output_fps > 0 else original_fps * N

        # Reassemble video
        frames_to_video(output_frames_dir, args.output, output_fps, audio_source=args.input)

        print(f"PROGRESS: Done! Output: {output_fps:.1f} FPS ({output_frame_idx} frames)")
        print(f"OUTPUT: {args.output}")
        return 0

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def interpolate_gimm(args):
    """Run GIMM-VFI interpolation."""
    import torch
    import numpy as np
    import cv2
    from PIL import Image

    script_dir = os.path.dirname(os.path.abspath(__file__))
    gimm_dir = os.path.join(script_dir, "GIMM-VFI")
    gimm_src_path = os.path.join(gimm_dir, "src")

    if gimm_src_path not in sys.path:
        sys.path.insert(0, gimm_src_path)

    from models import create_model
    from utils.setup import single_setup

    print(f"PROGRESS: Loading GIMM-VFI model ({args.variant})...")

    # Model configurations
    GIMM_CONFIGS = {
        "GIMM-VFI-R (RAFT)": ("configs/gimmvfi/gimmvfi_r_arb.yaml", "pretrained_ckpt/gimmvfi_r_arb.pt"),
        "GIMM-VFI-R-P (RAFT+Perceptual)": ("configs/gimmvfi/gimmvfi_r_arb.yaml", "pretrained_ckpt/gimmvfi_r_arb_lpips.pt"),
        "GIMM-VFI-F (FlowFormer)": ("configs/gimmvfi/gimmvfi_f_arb.yaml", "pretrained_ckpt/gimmvfi_f_arb.pt"),
        "GIMM-VFI-F-P (FlowFormer+Perceptual)": ("configs/gimmvfi/gimmvfi_f_arb.yaml", "pretrained_ckpt/gimmvfi_f_arb_lpips.pt"),
    }

    config_file, ckpt_file = GIMM_CONFIGS.get(args.variant, GIMM_CONFIGS["GIMM-VFI-R-P (RAFT+Perceptual)"])

    if args.config:
        config_file = args.config
    if args.checkpoint:
        ckpt_file = args.checkpoint

    # Setup config
    import argparse as ap
    abs_config_file = os.path.join(gimm_dir, config_file) if not os.path.isabs(config_file) else config_file
    setup_args = ap.Namespace(eval=True, resume=False, seed=0, model_config=abs_config_file)
    config = single_setup(setup_args, extra_args=[])

    # Create model (need to be in GIMM-VFI dir for RAFT paths)
    original_cwd = os.getcwd()
    try:
        os.chdir(gimm_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = create_model(config.arch)
        model = model.to(device)
    finally:
        os.chdir(original_cwd)

    # Load checkpoint
    abs_ckpt_file = os.path.join(gimm_dir, ckpt_file) if not os.path.isabs(ckpt_file) else ckpt_file
    ckpt = torch.load(abs_ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    print(f"PROGRESS: Model loaded, extracting frames...")

    # Create temp directories
    temp_dir = tempfile.mkdtemp(prefix="gimm_interp_")
    input_frames_dir = os.path.join(temp_dir, "input_frames")
    output_frames_dir = os.path.join(temp_dir, "output_frames")
    os.makedirs(input_frames_dir, exist_ok=True)
    os.makedirs(output_frames_dir, exist_ok=True)

    # Input padder class
    import torch.nn.functional as F

    class InputPadder:
        def __init__(self, dims, divisor=16):
            self.ht, self.wd = dims[-2:]
            pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
            pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
            self._pad = [0, pad_wd, 0, pad_ht]

        def pad(self, *inputs):
            if len(inputs) == 1:
                return F.pad(inputs[0], self._pad, mode="replicate")
            return [F.pad(x, self._pad, mode="replicate") for x in inputs]

        def unpad(self, *inputs):
            if len(inputs) == 1:
                return self._unpad(inputs[0])
            return [self._unpad(x) for x in inputs]

        def _unpad(self, x):
            ht, wd = x.shape[-2:]
            c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
            return x[..., c[0]:c[1], c[2]:c[3]]

    try:
        # Extract frames
        frame_paths, original_fps = extract_video_frames(args.input, input_frames_dir)
        print(f"PROGRESS: Extracted {len(frame_paths)} frames at {original_fps:.2f} FPS")

        if len(frame_paths) < 2:
            print("ERROR: Video must have at least 2 frames")
            return 1

        # Process frame pairs
        N = args.factor
        total_pairs = len(frame_paths) - 1
        output_frame_idx = 0
        ds_scale = args.ds_scale

        def load_image(img_path):
            img = Image.open(img_path)
            raw_img = np.array(img.convert("RGB"))
            img_tensor = torch.from_numpy(raw_img.copy()).permute(2, 0, 1) / 255.0
            return img_tensor.to(torch.float).unsqueeze(0)

        for pair_idx in range(total_pairs):
            progress_pct = int((pair_idx / total_pairs) * 100)
            print(f"PROGRESS: Interpolating pair {pair_idx + 1}/{total_pairs} ({progress_pct}%)")

            I0 = load_image(frame_paths[pair_idx]).to(device)
            I2 = load_image(frame_paths[pair_idx + 1]).to(device)

            padder = InputPadder(I0.shape, 32)
            I0_pad, I2_pad = padder.pad(I0, I2)

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
                    (model.sample_coord_input(batch_size, s_shape, [1 / N * i], device=xs.device, upsample_ratio=ds_scale), None)
                    for i in range(1, N)
                ]
                timesteps = [i * 1 / N * torch.ones(batch_size).to(xs.device).to(torch.float) for i in range(1, N)]
                outputs = model(xs, coord_inputs, t=timesteps, ds_factor=ds_scale)
                out_frames = [padder.unpad(im) for im in outputs["imgt_pred"]]

            for frame_tensor in out_frames:
                frame_np = (frame_tensor[0].cpu().numpy().transpose(1, 2, 0) * 255.0)
                frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_frames_dir, f"{output_frame_idx:05d}.png"), frame_bgr)
                output_frame_idx += 1

            # Save second frame
            frame_np = (padder.unpad(I2_pad)[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_frames_dir, f"{output_frame_idx:05d}.png"), frame_bgr)
            output_frame_idx += 1

            if pair_idx % 10 == 0:
                torch.cuda.empty_cache()

        print(f"PROGRESS: Encoding output video...")

        # Calculate output FPS
        output_fps = args.output_fps if args.output_fps > 0 else original_fps * N

        # Reassemble video
        frames_to_video(output_frames_dir, args.output, output_fps, audio_source=args.input)

        print(f"PROGRESS: Done! Output: {output_fps:.1f} FPS ({output_frame_idx} frames)")
        print(f"OUTPUT: {args.output}")
        return 0

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Video Frame Interpolation")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--model-type", choices=["bim", "gimm"], default="bim", help="Model type")
    parser.add_argument("--variant", default="GIMM-VFI-R-P (RAFT+Perceptual)", help="GIMM-VFI variant")
    parser.add_argument("--checkpoint", default="", help="Custom checkpoint path")
    parser.add_argument("--config", default="", help="Custom config path (GIMM only)")
    parser.add_argument("--factor", type=int, default=2, help="Interpolation factor")
    parser.add_argument("--pyr-level", type=int, default=0, help="Pyramid level (BiM only, 0=auto)")
    parser.add_argument("--ds-scale", type=float, default=1.0, help="DS scale (GIMM only)")
    parser.add_argument("--output-fps", type=float, default=0, help="Output FPS (0=auto)")
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

    if args.model_type == "bim":
        return interpolate_bim(args)
    else:
        return interpolate_gimm(args)


if __name__ == "__main__":
    sys.exit(main())
