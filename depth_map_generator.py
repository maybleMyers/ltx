#!/usr/bin/env python3
"""
Depth Map Generator for LTX-2 IC-LoRA Depth Control

Generates depth maps from images or videos using ZoeDepth for use with
LTX-2's depth control IC-LoRA. Supports batch processing and video output.

Usage:
    # Generate depth from a single image
    python depth_map_generator.py --input image.jpg --output depth.png

    # Generate depth video from input video
    python depth_map_generator.py --input video.mp4 --output depth_video.mp4

    # Generate depth from image, output as video (repeated frames)
    python depth_map_generator.py --input image.jpg --output depth.mp4 --num-frames 121
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def log(message: str):
    """Print message and flush immediately for real-time output."""
    print(message, flush=True)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DepthEstimator:
    """
    Depth estimation using ZoeDepth for IC-LoRA depth control.

    Uses the Intel/zoedepth-nyu-kitti model for monocular depth estimation.
    Provides lazy loading and automatic cleanup.
    """

    def __init__(self, device: torch.device | None = None, model_name: str = "Intel/zoedepth-nyu-kitti"):
        self.device = device or get_device()
        self.model_name = model_name
        self.model = None

    def load(self):
        """Lazy-load ZoeDepth model from HuggingFace."""
        if self.model is None:
            log(f">>> Loading depth model: {self.model_name}...")
            try:
                from transformers import pipeline
                self.model = pipeline(
                    "depth-estimation",
                    model=self.model_name,
                    device=0 if self.device.type == "cuda" else -1,
                )
                log(">>> Depth model loaded successfully")
            except ImportError:
                raise ImportError(
                    "Depth estimation requires transformers>=4.35.0. "
                    "Install with: pip install transformers>=4.35.0"
                )

    def estimate_pil(self, image: Image.Image) -> np.ndarray:
        """
        Estimate depth from a PIL Image.

        Args:
            image: PIL Image (RGB)

        Returns:
            Depth array [H, W] normalized to [0, 255] uint8
        """
        self.load()

        # Run depth estimation
        result = self.model(image)
        depth_pil = result["depth"]

        # Convert to numpy and normalize to 0-255
        depth_np = np.array(depth_pil).astype(np.float32)
        depth_min, depth_max = depth_np.min(), depth_np.max()
        if depth_max > depth_min:
            depth_np = (depth_np - depth_min) / (depth_max - depth_min) * 255.0
        else:
            depth_np = np.zeros_like(depth_np)

        return depth_np.astype(np.uint8)

    def estimate_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """
        Estimate depth from a torch tensor.

        Args:
            image: Image tensor [H, W, C] in range [0, 1] or [0, 255]

        Returns:
            Depth tensor [H, W] normalized to [0, 1]
        """
        # Convert to PIL
        if image.max() <= 1.0:
            image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        else:
            image_np = image.cpu().numpy().astype(np.uint8)

        pil_image = Image.fromarray(image_np)
        depth_np = self.estimate_pil(pil_image)

        return torch.from_numpy(depth_np.astype(np.float32) / 255.0)

    def unload(self):
        """Free model memory."""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log(">>> Depth model unloaded")


def load_video_frames(video_path: str, max_frames: int | None = None) -> list[Image.Image]:
    """
    Load video frames as PIL Images.

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to load (None = all)

    Returns:
        List of PIL Images
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("Video loading requires opencv-python. Install with: pip install opencv-python")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = min(frame_count, max_frames) if max_frames else frame_count

    log(f">>> Loading {total_frames} frames from video...")
    for _ in tqdm(range(total_frames), desc="Loading frames"):
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames


def get_video_fps(video_path: str) -> float:
    """Get the FPS of a video file."""
    try:
        import cv2
    except ImportError:
        return 24.0  # Default fallback

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 24.0


def save_depth_image(depth: np.ndarray, output_path: str, colorize: bool = False):
    """
    Save depth map as an image.

    Args:
        depth: Depth array [H, W] uint8
        output_path: Output file path
        colorize: Whether to apply colormap (True) or save as grayscale (False)
    """
    if colorize:
        import cv2
        depth_colored = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        Image.fromarray(depth_colored).save(output_path)
    else:
        # Save as RGB grayscale (same value in all channels for VAE compatibility)
        depth_rgb = np.stack([depth, depth, depth], axis=-1)
        Image.fromarray(depth_rgb).save(output_path)


def save_depth_video(
    depth_frames: list[np.ndarray],
    output_path: str,
    fps: float = 24.0,
    colorize: bool = False,
):
    """
    Save depth maps as a video.

    Args:
        depth_frames: List of depth arrays [H, W] uint8
        output_path: Output video path
        fps: Frames per second
        colorize: Whether to apply colormap
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("Video saving requires opencv-python. Install with: pip install opencv-python")

    if not depth_frames:
        raise ValueError("No depth frames to save")

    height, width = depth_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    log(f">>> Saving depth video ({len(depth_frames)} frames)...")
    for depth in tqdm(depth_frames, desc="Saving video"):
        if colorize:
            frame = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        else:
            # Grayscale as RGB
            frame = cv2.cvtColor(np.stack([depth, depth, depth], axis=-1), cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()
    log(f">>> Depth video saved: {output_path}")


def generate_depth_from_image(
    input_path: str,
    output_path: str,
    colorize: bool = False,
    num_frames: int | None = None,
    fps: float = 24.0,
    resize: tuple[int, int] | None = None,
) -> str:
    """
    Generate depth map from a single image.

    Args:
        input_path: Path to input image
        output_path: Path to output (image or video)
        colorize: Whether to apply colormap visualization
        num_frames: If set, output as video with this many frames
        fps: Video FPS (only used if num_frames is set)
        resize: Optional (width, height) to resize output

    Returns:
        Path to the output file
    """
    estimator = DepthEstimator()

    try:
        # Load and process image
        image = Image.open(input_path).convert("RGB")
        if resize:
            image = image.resize(resize, Image.Resampling.LANCZOS)

        log(f">>> Estimating depth for image: {input_path}")
        depth = estimator.estimate_pil(image)

        # Save as image or video
        if num_frames and num_frames > 1:
            # Output as video with repeated frames
            depth_frames = [depth] * num_frames
            save_depth_video(depth_frames, output_path, fps=fps, colorize=colorize)
        else:
            save_depth_image(depth, output_path, colorize=colorize)
            log(f">>> Depth image saved: {output_path}")

        return output_path

    finally:
        estimator.unload()


def generate_depth_from_video(
    input_path: str,
    output_path: str,
    colorize: bool = False,
    max_frames: int | None = None,
    resize: tuple[int, int] | None = None,
) -> str:
    """
    Generate depth video from input video.

    Args:
        input_path: Path to input video
        output_path: Path to output video
        colorize: Whether to apply colormap visualization
        max_frames: Maximum frames to process (None = all)
        resize: Optional (width, height) to resize output

    Returns:
        Path to the output file
    """
    estimator = DepthEstimator()

    try:
        # Load video frames
        frames = load_video_frames(input_path, max_frames=max_frames)
        fps = get_video_fps(input_path)

        # Resize frames if needed
        if resize:
            frames = [f.resize(resize, Image.Resampling.LANCZOS) for f in frames]

        # Estimate depth for each frame
        log(f">>> Estimating depth for {len(frames)} frames...")
        depth_frames = []
        total_frames = len(frames)
        for i, frame in enumerate(frames):
            depth = estimator.estimate_pil(frame)
            depth_frames.append(depth)
            # Log progress every 10 frames or at the end
            if (i + 1) % 10 == 0 or (i + 1) == total_frames:
                pct = int((i + 1) / total_frames * 100)
                log(f">>> Depth estimation: {pct}% ({i + 1}/{total_frames} frames)")

        # Save as video
        save_depth_video(depth_frames, output_path, fps=fps, colorize=colorize)
        return output_path

    finally:
        estimator.unload()


def main():
    parser = argparse.ArgumentParser(
        description="Generate depth maps from images or videos for LTX-2 IC-LoRA depth control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate depth from image
  python depth_map_generator.py --input photo.jpg --output depth.png

  # Generate depth video from video
  python depth_map_generator.py --input video.mp4 --output depth_video.mp4

  # Generate colorized depth visualization
  python depth_map_generator.py --input photo.jpg --output depth_colored.png --colorize

  # Generate depth video from image (repeated frames)
  python depth_map_generator.py --input photo.jpg --output depth.mp4 --num-frames 121 --fps 24
        """
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input image or video file"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output depth map (image or video)"
    )
    parser.add_argument(
        "--colorize",
        action="store_true",
        help="Apply colormap visualization (default: grayscale RGB)"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="For image input: output as video with this many frames"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Video FPS (default: 24.0)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="For video input: maximum frames to process"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Resize output width"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Resize output height"
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        log(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Determine resize
    resize = None
    if args.width and args.height:
        resize = (args.width, args.height)

    # Determine input type by extension
    input_ext = Path(args.input).suffix.lower()
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}

    log(f">>> Starting depth map generation")
    log(f">>> Input: {args.input}")
    log(f">>> Output: {args.output}")

    if input_ext in video_extensions:
        # Video input
        generate_depth_from_video(
            input_path=args.input,
            output_path=args.output,
            colorize=args.colorize,
            max_frames=args.max_frames,
            resize=resize,
        )
    else:
        # Image input
        generate_depth_from_image(
            input_path=args.input,
            output_path=args.output,
            colorize=args.colorize,
            num_frames=args.num_frames,
            fps=args.fps,
            resize=resize,
        )

    log(">>> Done!")


if __name__ == "__main__":
    main()
