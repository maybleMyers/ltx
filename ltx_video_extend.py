#!/usr/bin/env python3
"""
LTX-2 Video Extension Script (Wan2GP-style Implementation)

Extends videos using the VideoConditionByLatentIndex conditioning approach
from Wan2GP's ti2vid_two_stages.py pipeline. This approach:
1. Encodes preserved video frames to latent space
2. Applies conditioning at specific latent indices (strength=1.0 → denoise_mask=0)
3. Uses denoise_audio_video which properly handles clean_latent blending
4. Results in seamless video extension with proper frame preservation

Uses the backend model loader and offloading infrastructure from ltx_generate_video.py.
"""

import argparse
import gc
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Import from ltx_generate_video for backend infrastructure
from ltx_generate_video import (
    LTXVideoGeneratorWithOffloading,
    OOMRetryState,
    oom_retry_wrapper,
    synchronize_and_cleanup,
    phase_barrier,
    apply_loras_chunked_gpu,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_GEMMA_ROOT,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_SPATIAL_UPSAMPLER_PATH,
    DEFAULT_DISTILLED_LORA_PATH,
)

# Import cleanup_memory from helpers (canonical source)
from ltx_pipelines.utils.helpers import cleanup_memory
from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE

# Import LTX-2 core components
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import CFGGuider
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.conditioning import VideoConditionByLatentIndex, AudioConditionByLatent
from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.model.audio_vae import AudioProcessor, decode_audio as vae_decode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.text_encoders.gemma import encode_text
from ltx_core.types import LatentState, VideoPixelShape, VideoLatentShape, AudioLatentShape
from ltx_core.tools import VideoLatentTools, AudioLatentTools

from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.block_swap import (
    enable_block_swap,
    enable_block_swap_with_activation_offload,
    offload_all_blocks,
    enable_text_encoder_block_swap,
)
from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMA_VALUES, DEFAULT_LORA_STRENGTH
from ltx_pipelines.utils.helpers import (
    denoise_audio_video,
    euler_denoising_loop,
    guider_denoising_func,
    simple_denoising_func,
)
from ltx_pipelines.utils.media_io import decode_audio_from_file, encode_video
from ltx_pipelines.utils.types import PipelineComponents


def resolve_path(path: str) -> str:
    """Resolve a path, expanding user and making absolute."""
    if not path:
        return path
    return os.path.abspath(os.path.expanduser(path))


class LoraAction(argparse.Action):
    """Parse LoRA arguments: PATH [STRENGTH]"""
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) > 2:
            msg = f"{option_string} accepts at most 2 arguments (PATH and optional STRENGTH), got {len(values)} values"
            raise argparse.ArgumentError(self, msg)

        path = values[0]
        strength_str = values[1] if len(values) > 1 else str(DEFAULT_LORA_STRENGTH)

        resolved_path = resolve_path(path)
        strength = float(strength_str)

        current = getattr(namespace, self.dest) or []
        current.append(LoraPathStrengthAndSDOps(resolved_path, strength, LTXV_LORA_COMFY_RENAMING_MAP))
        setattr(namespace, self.dest, current)


@torch.inference_mode()
def extend_video(
    generator: LTXVideoGeneratorWithOffloading,
    input_video_path: str,
    prompt: str,
    negative_prompt: str = "",
    seed: int = 42,
    extend_seconds: float = 5.0,
    num_inference_steps: int = 30,
    cfg_guidance_scale: float = 3.0,
    preserve_strength: float = 1.0,
    skip_stage2: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Extend a video using Wan2GP-style VideoConditionByLatentIndex conditioning.

    This matches the approach in Wan2GP's ti2vid_two_stages.py:
    - Preserved frames are injected as conditioning with strength=1.0
    - This sets denoise_mask=0 and clean_latent=encoded_video for those frames
    - post_process_latent blends denoised output with clean_latent based on mask
    - Result: preserved frames stay pristine, new frames are generated

    Args:
        generator: LTXVideoGeneratorWithOffloading instance
        input_video_path: Path to video to extend
        prompt: Text prompt for generation
        negative_prompt: Negative prompt
        seed: Random seed
        extend_seconds: Seconds of video to generate after input
        num_inference_steps: Denoising steps
        cfg_guidance_scale: CFG scale
        preserve_strength: Strength for preserved frames (1.0 = fully frozen)
        skip_stage2: Skip stage 2 refinement

    Returns:
        Tuple of (video_tensor [F, H, W, C], audio_tensor or None)
    """
    device = generator.device
    dtype = generator.dtype

    print("=" * 60)
    print("Video Extension (Wan2GP-style Conditioning)")
    print("=" * 60)

    # =========================================================================
    # Step 1: Load and analyze input video
    # =========================================================================
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_video_path}")

    total_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_duration = total_input_frames / input_fps

    print(f">>> Input video: {total_input_frames} frames at {input_fps:.1f} fps")
    print(f">>> Resolution: {video_width}x{video_height}")
    print(f">>> Duration: {input_duration:.2f}s")

    # Calculate output parameters
    output_fps = input_fps
    output_duration = input_duration + extend_seconds
    output_frames = int(output_duration * output_fps)

    # Ensure output frames is 8n+1 format for VAE
    output_frames = ((output_frames - 1) // 8) * 8 + 1

    # Also ensure input frames to preserve is 8n+1
    preserve_pixel_frames = ((total_input_frames - 1) // 8) * 8 + 1
    preserve_pixel_frames = min(preserve_pixel_frames, total_input_frames)

    print(f">>> Output: {output_frames} frames ({output_frames / output_fps:.2f}s)")
    print(f">>> Preserving: {preserve_pixel_frames} frames")
    print(f">>> Generating: {output_frames - preserve_pixel_frames} new frames")

    # Determine resolution
    if generator.one_stage:
        stage1_width = video_width
        stage1_height = video_height
    else:
        stage1_width = video_width // 2
        stage1_height = video_height // 2

    # Ensure divisible by 32
    stage1_width = (stage1_width // 32) * 32
    stage1_height = (stage1_height // 32) * 32

    print(f">>> Stage 1 resolution: {stage1_width}x{stage1_height}")

    # =========================================================================
    # Step 2: Load video frames
    # =========================================================================
    print(">>> Loading video frames...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    input_frames = []
    for _ in range(preserve_pixel_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (stage1_width, stage1_height), interpolation=cv2.INTER_LANCZOS4)
        input_frames.append(frame)
    cap.release()

    print(f">>> Loaded {len(input_frames)} frames")

    # Convert to tensor [1, C, F, H, W] normalized to [-1, 1]
    frames_tensor = torch.from_numpy(np.stack(input_frames)).float() / 255.0
    frames_tensor = frames_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, F, H, W]
    frames_tensor = frames_tensor * 2.0 - 1.0

    # =========================================================================
    # Step 3: Extract audio from input video
    # =========================================================================
    print(">>> Extracting audio...")
    audio_waveform = None
    audio_sample_rate = None
    try:
        waveform, sample_rate = decode_audio_from_file(input_video_path, device)
        if waveform is not None:
            audio_waveform = waveform
            audio_sample_rate = sample_rate
            print(f">>> Audio: {waveform.shape}, {sample_rate}Hz")
    except Exception as e:
        print(f">>> No audio: {e}")

    # =========================================================================
    # Step 4: Encode video to latent space
    # =========================================================================
    print(">>> Encoding video to latent space...")
    video_encoder = generator.stage_1_model_ledger.video_encoder()
    encoder_dtype = next(video_encoder.parameters()).dtype

    # Get VAE temporal scale factor
    time_scale_factor = 8  # LTX-2 uses 8x temporal compression

    # Encode in chunks if needed
    chunk_pixel_frames = 65  # 8*8+1
    total_pixel_frames = frames_tensor.shape[2]

    latent_chunks = []
    with torch.no_grad():
        for start_frame in range(0, total_pixel_frames, chunk_pixel_frames - 1):
            end_frame = min(start_frame + chunk_pixel_frames, total_pixel_frames)
            chunk = frames_tensor[:, :, start_frame:end_frame, :, :]

            # Pad to valid 8n+1 if needed
            actual_frames = chunk.shape[2]
            if actual_frames > 1:
                n = (actual_frames - 1 + 7) // 8
                target_frames = 8 * n + 1
                if actual_frames < target_frames:
                    pad_frames = target_frames - actual_frames
                    last_frame = chunk[:, :, -1:, :, :].expand(-1, -1, pad_frames, -1, -1)
                    chunk = torch.cat([chunk, last_frame], dim=2)

            chunk_latent = video_encoder(chunk.to(device=device, dtype=encoder_dtype))
            chunk_latent = chunk_latent.to(dtype=dtype)

            # Skip first frame for overlapping chunks
            if len(latent_chunks) > 0:
                chunk_latent = chunk_latent[:, :, 1:, :, :]

            latent_chunks.append(chunk_latent)
            torch.cuda.empty_cache()

    preserved_video_latent = torch.cat(latent_chunks, dim=2)
    preserve_latent_frames = preserved_video_latent.shape[2]
    print(f">>> Encoded latent: {preserved_video_latent.shape}")

    del frames_tensor, latent_chunks
    cleanup_memory()

    # =========================================================================
    # Step 5: Encode audio to latent (if present)
    # =========================================================================
    preserved_audio_latent = None
    if audio_waveform is not None:
        print(">>> Encoding audio to latent space...")
        audio_encoder = generator.stage_1_model_ledger.audio_encoder()
        audio_processor = AudioProcessor(
            sample_rate=audio_encoder.sample_rate,
            mel_bins=audio_encoder.mel_bins,
            mel_hop_length=audio_encoder.mel_hop_length,
            n_fft=audio_encoder.n_fft,
        ).to(device)

        # Prepare waveform
        if audio_waveform.dim() == 3:
            num_frames_audio, channels, samples_per_frame = audio_waveform.shape
            audio_waveform = audio_waveform.permute(1, 0, 2).reshape(channels, -1).unsqueeze(0)
        elif audio_waveform.dim() == 2:
            audio_waveform = audio_waveform.unsqueeze(0)

        mel_spectrogram = audio_processor.waveform_to_mel(
            audio_waveform.to(dtype=torch.float32),
            waveform_sample_rate=audio_sample_rate or AUDIO_SAMPLE_RATE
        )

        with torch.no_grad():
            preserved_audio_latent = audio_encoder(mel_spectrogram.to(dtype=torch.float32).to(device))
            preserved_audio_latent = preserved_audio_latent.to(dtype=dtype)

        print(f">>> Audio latent: {preserved_audio_latent.shape}")

        del audio_encoder, audio_processor
        cleanup_memory()

    del video_encoder
    cleanup_memory()

    # =========================================================================
    # Step 6: Load text encoder and encode prompts
    # =========================================================================
    print(">>> Encoding prompts...")

    if generator.enable_text_encoder_block_swap:
        original_device = generator.stage_1_model_ledger.device
        generator.stage_1_model_ledger.device = torch.device("cpu")
        text_encoder = generator.stage_1_model_ledger.text_encoder()
        generator.stage_1_model_ledger.device = original_device
        text_encoder_block_swap = enable_text_encoder_block_swap(
            text_encoder,
            blocks_in_memory=generator.text_encoder_blocks_in_memory,
            device=device,
        )
    else:
        text_encoder = generator.stage_1_model_ledger.text_encoder()

    if cfg_guidance_scale > 1.0:
        context_p, context_n = encode_text(text_encoder, prompts=[prompt, negative_prompt])
        v_context_p, a_context_p = context_p
        v_context_n, a_context_n = context_n
    else:
        context_p = encode_text(text_encoder, prompts=[prompt])[0]
        v_context_p, a_context_p = context_p
        v_context_n, a_context_n = None, None

    del text_encoder
    cleanup_memory()

    # =========================================================================
    # Step 7: Build conditioning list (Wan2GP-style)
    # =========================================================================
    print(">>> Building conditioning (Wan2GP-style)...")

    # Calculate output latent frames
    output_latent_frames = (output_frames - 1) // time_scale_factor + 1

    # Create conditioning for each preserved latent frame
    # VideoConditionByLatentIndex sets:
    #   - latent at that position = encoded frame
    #   - clean_latent at that position = encoded frame
    #   - denoise_mask at that position = 1.0 - strength (so strength=1.0 → mask=0)
    stage_1_conditionings = []

    for latent_idx in range(preserve_latent_frames):
        frame_latent = preserved_video_latent[:, :, latent_idx:latent_idx+1, :, :]
        stage_1_conditionings.append(
            VideoConditionByLatentIndex(
                latent=frame_latent,
                strength=preserve_strength,  # 1.0 = fully preserved (mask=0)
                latent_idx=latent_idx,
            )
        )

    print(f">>> Added {len(stage_1_conditionings)} frame conditionings (strength={preserve_strength})")

    # Audio conditioning (if present)
    audio_conditionings = []
    if preserved_audio_latent is not None:
        audio_conditionings.append(
            AudioConditionByLatent(
                latent=preserved_audio_latent,
                strength=preserve_strength,
            )
        )

    # =========================================================================
    # Step 8: Load transformer and run Stage 1 denoising
    # =========================================================================
    print(">>> Loading transformer...")

    block_swap_manager = None
    if generator.enable_dit_block_swap:
        from ltx_core.loader.sft_loader import SafetensorsStateDictLoader

        has_loras = hasattr(generator.stage_1_model_ledger, 'loras') and generator.stage_1_model_ledger.loras

        if has_loras:
            stage_1_ledger_no_lora = ModelLedger(
                dtype=generator.dtype,
                device=torch.device("cpu"),
                checkpoint_path=generator.stage_1_model_ledger.checkpoint_path,
                gemma_root_path=generator.stage_1_model_ledger.gemma_root_path,
                spatial_upsampler_path=generator.stage_1_model_ledger.spatial_upsampler_path,
                vae_path=generator.stage_1_model_ledger.vae_path,
                loras=(),
                fp8transformer=generator.stage_1_model_ledger.fp8transformer,
            )
            transformer = stage_1_ledger_no_lora.transformer()

            loras = generator.stage_1_model_ledger.loras
            print(f">>> Applying {len(loras)} LoRA(s)...")
            lora_loader = SafetensorsStateDictLoader()
            lora_state_dicts = []
            lora_strengths = []
            for lora in loras:
                lora_sd = lora_loader.load(lora.path, sd_ops=lora.sd_ops, device=torch.device("cpu"))
                lora_state_dicts.append(lora_sd)
                lora_strengths.append(lora.strength)

            apply_loras_chunked_gpu(
                model=transformer,
                lora_state_dicts=lora_state_dicts,
                lora_strengths=lora_strengths,
                gpu_device=device,
                dtype=dtype,
            )
            del lora_state_dicts
            cleanup_memory()
        else:
            original_device = generator.stage_1_model_ledger.device
            generator.stage_1_model_ledger.device = torch.device("cpu")
            transformer = generator.stage_1_model_ledger.transformer()
            generator.stage_1_model_ledger.device = original_device

        # Move non-block components to GPU
        transformer.velocity_model.patchify_proj.to(device)
        transformer.velocity_model.adaln_single.to(device)
        transformer.velocity_model.caption_projection.to(device)
        transformer.velocity_model.norm_out.to(device)
        transformer.velocity_model.proj_out.to(device)
        transformer.velocity_model.scale_shift_table = torch.nn.Parameter(
            transformer.velocity_model.scale_shift_table.to(device)
        )

        # Audio components
        for attr in ["audio_patchify_proj", "audio_adaln_single", "audio_caption_projection",
                     "audio_norm_out", "audio_proj_out"]:
            if hasattr(transformer.velocity_model, attr):
                getattr(transformer.velocity_model, attr).to(device)
        if hasattr(transformer.velocity_model, "audio_scale_shift_table"):
            transformer.velocity_model.audio_scale_shift_table = torch.nn.Parameter(
                transformer.velocity_model.audio_scale_shift_table.to(device)
            )

        # Cross-attention components
        for attr in ["av_ca_video_scale_shift_adaln_single", "av_ca_audio_scale_shift_adaln_single",
                     "av_ca_a2v_gate_adaln_single", "av_ca_v2a_gate_adaln_single"]:
            if hasattr(transformer.velocity_model, attr):
                getattr(transformer.velocity_model, attr).to(device)

        if getattr(generator, 'enable_activation_offload', False):
            block_swap_manager = enable_block_swap_with_activation_offload(
                transformer,
                blocks_in_memory=generator.dit_blocks_in_memory,
                device=device,
                verbose=True,
                temporal_chunk_size=getattr(generator, 'temporal_chunk_size', 0),
            )
        else:
            block_swap_manager = enable_block_swap(
                transformer,
                blocks_in_memory=generator.dit_blocks_in_memory,
                device=device,
            )
    else:
        transformer = generator.stage_1_model_ledger.transformer()

    # Create sigma schedule
    sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=device)

    # Initialize diffusion components
    generator_torch = torch.Generator(device=device).manual_seed(seed)
    noiser = GaussianNoiser(generator=generator_torch)
    stepper = EulerDiffusionStep()
    cfg_guider = CFGGuider(cfg_guidance_scale)

    # Create output shape
    stage_1_output_shape = VideoPixelShape(
        batch=1,
        frames=output_frames,
        height=stage1_height,
        width=stage1_width,
        fps=output_fps,
    )

    # Define denoising function
    if cfg_guidance_scale > 1.0:
        denoise_fn = guider_denoising_func(
            cfg_guider,
            v_context_p, v_context_n,
            a_context_p, a_context_n,
            transformer=transformer,
        )
    else:
        denoise_fn = simple_denoising_func(v_context_p, a_context_p, transformer)

    def stage1_denoising_loop(
        sigmas: torch.Tensor,
        video_state: LatentState,
        audio_state: LatentState,
        stepper: DiffusionStepProtocol,
        preview_tools=None,
        mask_context=None,
    ) -> tuple[LatentState, LatentState]:
        return euler_denoising_loop(
            sigmas=sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=stepper,
            denoise_fn=denoise_fn,
        )

    # Run Stage 1 denoising using denoise_audio_video (Wan2GP pattern)
    # This function:
    # 1. Creates initial state with output shape
    # 2. Applies conditionings (sets latent, clean_latent, denoise_mask)
    # 3. Runs noiser (adds noise only where mask > 0)
    # 4. Runs denoising loop (post_process_latent blends with clean_latent)
    print(f">>> Stage 1 denoising ({len(sigmas) - 1} steps)...")

    stage1_retry_state = OOMRetryState(
        stage="stage1_extension",
        original_blocks=generator.dit_blocks_in_memory if generator.enable_dit_block_swap else 0,
        current_blocks=generator.dit_blocks_in_memory if generator.enable_dit_block_swap else 0,
        min_blocks=2,
        original_ffn_chunk_size=generator.ffn_chunk_size,
        original_activation_offload=generator.enable_activation_offload,
        original_temporal_chunk_size=generator.temporal_chunk_size,
    )

    def stage1_denoise_call(**kwargs):
        return denoise_audio_video(**kwargs)

    stage1_kwargs = {
        'output_shape': stage_1_output_shape,
        'conditionings': stage_1_conditionings,
        'audio_conditionings': audio_conditionings if audio_conditionings else None,
        'noiser': noiser,
        'sigmas': sigmas,
        'stepper': stepper,
        'denoising_loop_fn': stage1_denoising_loop,
        'components': generator.pipeline_components,
        'dtype': dtype,
        'device': device,
    }

    (video_state, audio_state), block_swap_manager = oom_retry_wrapper(
        retry_state=stage1_retry_state,
        denoising_fn=stage1_denoise_call,
        transformer=transformer,
        generator_instance=generator,
        block_swap_manager=block_swap_manager,
        seed=seed,
        **stage1_kwargs
    )

    if video_state is None:
        raise RuntimeError("Stage 1 denoising failed")

    # Cleanup stage 1 transformer
    if block_swap_manager is not None:
        offload_all_blocks(transformer)
        transformer.velocity_model._block_swap_offloader = None
        transformer.velocity_model._blocks_ref = None
        block_swap_manager = None
    del transformer
    cleanup_memory()

    denoised_video_latent = video_state.latent
    denoised_audio_latent = audio_state.latent if audio_state else None

    print(f">>> Stage 1 complete: {denoised_video_latent.shape}")

    # =========================================================================
    # Step 9: Stage 2 refinement (if enabled)
    # =========================================================================
    if not skip_stage2 and not generator.one_stage:
        print(">>> Stage 2 refinement...")

        # Upsample latent
        video_encoder = generator.stage_1_model_ledger.video_encoder()
        spatial_upsampler = generator.stage_2_model_ledger.spatial_upsampler()

        upscaled_video_latent = upsample_video(
            latent=denoised_video_latent,
            video_encoder=video_encoder,
            upsampler=spatial_upsampler,
        )

        del spatial_upsampler
        cleanup_memory()

        # Re-encode preserved frames at full resolution for quality
        print(">>> Re-encoding preserved frames at full resolution...")
        stage2_width = stage1_width * 2
        stage2_height = stage1_height * 2

        cap = cv2.VideoCapture(input_video_path)
        full_res_frames = []
        for _ in range(preserve_pixel_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (stage2_width, stage2_height), interpolation=cv2.INTER_LANCZOS4)
            full_res_frames.append(frame)
        cap.release()

        # Encode full resolution frames
        encoder_dtype = next(video_encoder.parameters()).dtype
        full_res_tensor = torch.from_numpy(np.stack(full_res_frames)).float() / 255.0
        full_res_tensor = full_res_tensor.permute(3, 0, 1, 2).unsqueeze(0) * 2.0 - 1.0

        full_res_latent_chunks = []
        chunk_pixel_frames = 17  # Smaller chunks for high resolution
        with torch.no_grad():
            for start_frame in range(0, full_res_tensor.shape[2], chunk_pixel_frames - 1):
                end_frame = min(start_frame + chunk_pixel_frames, full_res_tensor.shape[2])
                chunk = full_res_tensor[:, :, start_frame:end_frame, :, :]

                actual_frames = chunk.shape[2]
                if actual_frames < 9:
                    pad_frames = 9 - actual_frames
                    last_frame = chunk[:, :, -1:, :, :].expand(-1, -1, pad_frames, -1, -1)
                    chunk = torch.cat([chunk, last_frame], dim=2)

                chunk_latent = video_encoder(chunk.to(device=device, dtype=encoder_dtype))
                chunk_latent = chunk_latent.to(dtype=dtype)

                if len(full_res_latent_chunks) > 0:
                    chunk_latent = chunk_latent[:, :, 1:, :, :]

                full_res_latent_chunks.append(chunk_latent)
                torch.cuda.empty_cache()

        full_res_latent = torch.cat(full_res_latent_chunks, dim=2)

        # Replace preserved frames in upscaled latent
        upscaled_video_latent[:, :, :preserve_latent_frames, :, :] = full_res_latent[:, :, :preserve_latent_frames, :, :]

        del full_res_frames, full_res_tensor, full_res_latent_chunks, full_res_latent, video_encoder
        cleanup_memory()

        # Build Stage 2 conditioning
        stage_2_conditionings = []
        for latent_idx in range(preserve_latent_frames):
            frame_latent = upscaled_video_latent[:, :, latent_idx:latent_idx+1, :, :]
            stage_2_conditionings.append(
                VideoConditionByLatentIndex(
                    latent=frame_latent.clone(),
                    strength=preserve_strength,
                    latent_idx=latent_idx,
                )
            )

        # Load Stage 2 transformer
        print(">>> Loading stage 2 transformer...")
        stage2_block_swap_manager = None
        if generator.enable_refiner_block_swap:
            from ltx_core.loader.sft_loader import SafetensorsStateDictLoader

            stage_2_ledger_no_lora = ModelLedger(
                dtype=generator.dtype,
                device=torch.device("cpu"),
                checkpoint_path=generator.stage_1_model_ledger.checkpoint_path,
                gemma_root_path=generator.stage_1_model_ledger.gemma_root_path,
                spatial_upsampler_path=generator.stage_1_model_ledger.spatial_upsampler_path,
                vae_path=generator.stage_1_model_ledger.vae_path,
                loras=(),
                fp8transformer=generator.stage_1_model_ledger.fp8transformer,
            )
            stage2_transformer = stage_2_ledger_no_lora.transformer()

            if hasattr(generator.stage_2_model_ledger, 'loras') and generator.stage_2_model_ledger.loras:
                loras = generator.stage_2_model_ledger.loras
                print(f">>> Applying {len(loras)} stage 2 LoRA(s)...")
                lora_loader = SafetensorsStateDictLoader()
                lora_state_dicts = []
                lora_strengths = []
                for lora in loras:
                    lora_sd = lora_loader.load(lora.path, sd_ops=lora.sd_ops, device=torch.device("cpu"))
                    lora_state_dicts.append(lora_sd)
                    lora_strengths.append(lora.strength)

                apply_loras_chunked_gpu(
                    model=stage2_transformer,
                    lora_state_dicts=lora_state_dicts,
                    lora_strengths=lora_strengths,
                    gpu_device=device,
                    dtype=dtype,
                )
                del lora_state_dicts
                cleanup_memory()

            # Move components to GPU
            stage2_transformer.velocity_model.patchify_proj.to(device)
            stage2_transformer.velocity_model.adaln_single.to(device)
            stage2_transformer.velocity_model.caption_projection.to(device)
            stage2_transformer.velocity_model.norm_out.to(device)
            stage2_transformer.velocity_model.proj_out.to(device)
            stage2_transformer.velocity_model.scale_shift_table = torch.nn.Parameter(
                stage2_transformer.velocity_model.scale_shift_table.to(device)
            )

            for attr in ["audio_patchify_proj", "audio_adaln_single", "audio_caption_projection",
                         "audio_norm_out", "audio_proj_out"]:
                if hasattr(stage2_transformer.velocity_model, attr):
                    getattr(stage2_transformer.velocity_model, attr).to(device)
            if hasattr(stage2_transformer.velocity_model, "audio_scale_shift_table"):
                stage2_transformer.velocity_model.audio_scale_shift_table = torch.nn.Parameter(
                    stage2_transformer.velocity_model.audio_scale_shift_table.to(device)
                )
            for attr in ["av_ca_video_scale_shift_adaln_single", "av_ca_audio_scale_shift_adaln_single",
                         "av_ca_a2v_gate_adaln_single", "av_ca_v2a_gate_adaln_single"]:
                if hasattr(stage2_transformer.velocity_model, attr):
                    getattr(stage2_transformer.velocity_model, attr).to(device)

            if getattr(generator, 'enable_activation_offload', False):
                stage2_block_swap_manager = enable_block_swap_with_activation_offload(
                    stage2_transformer,
                    blocks_in_memory=generator.refiner_blocks_in_memory,
                    device=device,
                    verbose=True,
                )
            else:
                stage2_block_swap_manager = enable_block_swap(
                    stage2_transformer,
                    blocks_in_memory=generator.refiner_blocks_in_memory,
                    device=device,
                )
        else:
            stage2_transformer = generator.stage_2_model_ledger.transformer()

        # Stage 2 sigma schedule
        stage2_sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=device)

        stage_2_output_shape = VideoPixelShape(
            batch=1,
            frames=output_frames,
            height=stage2_height,
            width=stage2_width,
            fps=output_fps,
        )

        # Stage 2 denoise function (no CFG)
        stage2_denoise_fn = simple_denoising_func(v_context_p, a_context_p, stage2_transformer)

        def stage2_denoising_loop(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: DiffusionStepProtocol,
            preview_tools=None,
            mask_context=None,
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=stage2_denoise_fn,
            )

        print(f">>> Stage 2 denoising ({len(stage2_sigmas) - 1} steps)...")

        stage2_retry_state = OOMRetryState(
            stage="stage2_extension",
            original_blocks=generator.refiner_blocks_in_memory if generator.enable_refiner_block_swap else 0,
            current_blocks=generator.refiner_blocks_in_memory if generator.enable_refiner_block_swap else 0,
            min_blocks=1,
            original_ffn_chunk_size=generator.ffn_chunk_size,
            original_activation_offload=generator.enable_activation_offload,
            original_temporal_chunk_size=generator.temporal_chunk_size,
        )

        stage2_kwargs = {
            'output_shape': stage_2_output_shape,
            'conditionings': stage_2_conditionings,
            'audio_conditionings': audio_conditionings if audio_conditionings else None,
            'noiser': noiser,
            'sigmas': stage2_sigmas,
            'stepper': stepper,
            'denoising_loop_fn': stage2_denoising_loop,
            'components': generator.pipeline_components,
            'dtype': dtype,
            'device': device,
            'noise_scale': stage2_sigmas[0].item(),
            'initial_video_latent': upscaled_video_latent,
            'initial_audio_latent': denoised_audio_latent,
        }

        (video_state, audio_state), stage2_block_swap_manager = oom_retry_wrapper(
            retry_state=stage2_retry_state,
            denoising_fn=stage1_denoise_call,  # Same function signature
            transformer=stage2_transformer,
            generator_instance=generator,
            block_swap_manager=stage2_block_swap_manager,
            seed=seed,
            **stage2_kwargs
        )

        if video_state is None:
            raise RuntimeError("Stage 2 denoising failed")

        # Cleanup
        if stage2_block_swap_manager is not None:
            offload_all_blocks(stage2_transformer)
            stage2_transformer.velocity_model._block_swap_offloader = None
            stage2_transformer.velocity_model._blocks_ref = None
        del stage2_transformer
        cleanup_memory()

        denoised_video_latent = video_state.latent
        denoised_audio_latent = audio_state.latent if audio_state else None

        print(f">>> Stage 2 complete: {denoised_video_latent.shape}")

    # =========================================================================
    # Step 10: Decode video and audio
    # =========================================================================
    print(">>> Decoding video...")

    if not skip_stage2 and not generator.one_stage:
        video_decoder = generator.stage_2_model_ledger.video_decoder()
    else:
        video_decoder = generator.stage_1_model_ledger.video_decoder()

    tiling_config = TilingConfig.default()
    decoded_chunks = []
    for chunk in vae_decode_video(denoised_video_latent, video_decoder, tiling_config):
        decoded_chunks.append(chunk.cpu())
        del chunk
    decoded_video = torch.cat(decoded_chunks, dim=0)  # [F, H, W, C]

    del video_decoder, decoded_chunks
    cleanup_memory()

    # Replace preserved frames with original quality pixels
    print(">>> Replacing preserved frames with original pixels...")
    out_h, out_w = decoded_video.shape[1], decoded_video.shape[2]

    cap = cv2.VideoCapture(input_video_path)
    original_frames = []
    for i in range(preserve_pixel_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        original_frames.append(frame)
    cap.release()

    if original_frames:
        original_pixels = torch.from_numpy(np.stack(original_frames))
        num_replace = min(len(original_frames), decoded_video.shape[0])
        decoded_video[:num_replace] = original_pixels[:num_replace]
        print(f">>> Replaced {num_replace} frames")

    # Decode audio
    decoded_audio = None
    if denoised_audio_latent is not None:
        print(">>> Decoding audio...")
        audio_decoder = generator.stage_1_model_ledger.audio_decoder()
        vocoder = generator.stage_1_model_ledger.vocoder()
        decoded_audio = vae_decode_audio(denoised_audio_latent, audio_decoder, vocoder)
        del audio_decoder, vocoder
        cleanup_memory()

    print(f">>> Done! Output: {decoded_video.shape}")

    return decoded_video, decoded_audio


def main():
    parser = argparse.ArgumentParser(description="LTX-2 Video Extension (Wan2GP-style)")

    # Input/output
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", default="extended_output.mp4", help="Output video path")
    parser.add_argument("--prompt", "-p", required=True, help="Text prompt")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt")

    # Generation settings
    parser.add_argument("--extend-seconds", type=float, default=5.0, help="Seconds to extend")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--steps", type=int, default=30, help="Denoising steps")
    parser.add_argument("--cfg", type=float, default=3.0, help="CFG guidance scale")
    parser.add_argument("--preserve-strength", type=float, default=1.0, help="Preservation strength (1.0 = fully frozen)")

    # Model paths
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH, help="Model checkpoint path")
    parser.add_argument("--gemma-root", default=DEFAULT_GEMMA_ROOT, help="Gemma text encoder path")
    parser.add_argument("--spatial-upsampler", default=DEFAULT_SPATIAL_UPSAMPLER_PATH, help="Spatial upsampler path")
    parser.add_argument("--distilled-lora", default=DEFAULT_DISTILLED_LORA_PATH, help="Distilled LoRA path")

    # Memory optimization
    parser.add_argument("--one-stage", action="store_true", help="Skip stage 2")
    parser.add_argument("--skip-stage2", action="store_true", help="Skip stage 2 refinement")
    parser.add_argument("--dit-block-swap", action="store_true", help="Enable DiT block swapping")
    parser.add_argument("--dit-blocks", type=int, default=22, help="DiT blocks in memory")
    parser.add_argument("--text-encoder-block-swap", action="store_true", help="Enable text encoder block swapping")
    parser.add_argument("--text-encoder-blocks", type=int, default=6, help="Text encoder blocks in memory")
    parser.add_argument("--refiner-block-swap", action="store_true", help="Enable refiner block swapping")
    parser.add_argument("--refiner-blocks", type=int, default=22, help="Refiner blocks in memory")
    parser.add_argument("--activation-offload", action="store_true", help="Enable activation offloading")

    # LoRA support
    parser.add_argument(
        "--lora",
        dest="loras",
        action=LoraAction,
        nargs="+",
        metavar=("PATH", "STRENGTH"),
        default=[],
        help="User LoRA for stage 1 (base generation): path and optional strength. Can be repeated.",
    )
    parser.add_argument(
        "--stage2-lora",
        dest="stage2_loras",
        action=LoraAction,
        nargs="+",
        metavar=("PATH", "STRENGTH"),
        default=[],
        help="User LoRA for stage 2 (refinement) only: path and optional strength. Can be repeated.",
    )

    args = parser.parse_args()

    # Build distilled LoRA list
    distilled_lora = []
    if args.distilled_lora and os.path.exists(args.distilled_lora):
        distilled_lora = [LoraPathStrengthAndSDOps(path=args.distilled_lora, strength=1.0, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)]

    # Get user LoRAs
    user_loras = args.loras or []
    stage2_loras = args.stage2_loras or []

    if user_loras:
        print(f">>> Stage 1 LoRAs: {[l.path for l in user_loras]}")
    if stage2_loras:
        print(f">>> Stage 2 LoRAs: {[l.path for l in stage2_loras]}")

    # Create generator
    generator = LTXVideoGeneratorWithOffloading(
        checkpoint_path=args.checkpoint,
        distilled_lora=distilled_lora,
        spatial_upsampler_path=args.spatial_upsampler,
        gemma_root=args.gemma_root,
        loras=user_loras,
        stage2_loras=stage2_loras,
        enable_dit_block_swap=args.dit_block_swap,
        dit_blocks_in_memory=args.dit_blocks,
        enable_text_encoder_block_swap=args.text_encoder_block_swap,
        text_encoder_blocks_in_memory=args.text_encoder_blocks,
        enable_refiner_block_swap=args.refiner_block_swap,
        refiner_blocks_in_memory=args.refiner_blocks,
        enable_activation_offload=args.activation_offload,
        one_stage=args.one_stage,
    )

    # Run extension
    video, audio = extend_video(
        generator=generator,
        input_video_path=args.input,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        extend_seconds=args.extend_seconds,
        num_inference_steps=args.steps,
        cfg_guidance_scale=args.cfg,
        preserve_strength=args.preserve_strength,
        skip_stage2=args.skip_stage2 or args.one_stage,
    )

    # Save output
    print(f">>> Saving to {args.output}...")

    # Get fps from input video
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    encode_video(
        video=iter([video]),
        fps=fps,
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=args.output,
        video_chunks_number=1,
    )

    print(f">>> Done! Saved to {args.output}")


if __name__ == "__main__":
    main()
