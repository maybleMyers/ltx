#!/usr/bin/env python3
"""
LTX-2 Video Generation Script

A comprehensive CLI tool for generating videos using the LTX-2 model with:
- Two-stage pipeline (low-res generation + hi-res refinement)
- Joint audio-video generation
- Intelligent CPU/GPU offloading
- Block swapping for memory efficiency
- Image conditioning (I2V mode)
- LoRA support
- FP8 quantization

Based on the LTX-2 ti2vid_two_stages pipeline with advanced memory management
features inspired by Kandinsky5.
"""

import argparse
import gc
import logging
import os
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

import torch
from tqdm import tqdm

# Import LTX-2 components
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import CFGGuider, STGGuider
from ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import AudioProcessor, decode_audio as vae_decode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.text_encoders.gemma import encode_text
from ltx_core.types import LatentState, VideoPixelShape

from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.block_swap import (
    enable_block_swap,
    enable_block_swap_with_activation_offload,
    offload_all_blocks,
    enable_text_encoder_block_swap,
    offload_all_text_encoder_blocks,
)
from ltx_pipelines.utils.custom_offloading_utils import clean_memory_on_device
from ltx_pipelines.utils.constants import (
    AUDIO_SAMPLE_RATE,
    DEFAULT_CFG_GUIDANCE_SCALE,
    DEFAULT_FRAME_RATE,
    DEFAULT_LORA_STRENGTH,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NUM_FRAMES,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_SEED,
    DISTILLED_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMA_VALUES,
)
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    cleanup_memory,
    create_per_step_adain_norm_fn,
    create_per_step_stat_norm_fn,
    denoise_audio_video,
    euler_denoising_loop,
    generate_enhanced_prompt,
    get_device,
    guider_denoising_func,
    image_conditionings_by_adding_guiding_latent,
    image_conditionings_by_replacing_latent,
    simple_denoising_func,
)
from ltx_pipelines.utils.media_io import decode_audio_from_file, encode_video, load_video_conditioning
from ltx_pipelines.utils.types import PipelineComponents


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_GEMMA_ROOT = "./gemma-3-12b-it-qat-q4_0-unquantized"
DEFAULT_CHECKPOINT_PATH = "./weights/ltx-2-19b-dev.safetensors"
DEFAULT_SPATIAL_UPSAMPLER_PATH = "./weights/ltx-2-spatial-upscaler-x2-1.0.safetensors"
DEFAULT_DISTILLED_LORA_PATH = "./weights/ltx-2-19b-distilled-lora-384.safetensors"


# =============================================================================
# LTX-V RGB Factors for Latent Preview (128 channels)
# =============================================================================
# These factors enable fast latent-to-RGB conversion without VAE decoding.
# Source: Wan2GP/shared/RGB_factors.py (computed for LTXv model)
LATENT_RGB_FACTORS_LTXV = [
    [1.1202e-02, -6.3815e-04, -1.0021e-02],
    [8.6031e-02, 6.5813e-02, 9.5409e-04],
    [-1.2576e-02, -7.5734e-03, -4.0528e-03],
    [9.4063e-03, -2.1688e-03, 2.6093e-03],
    [3.7636e-03, 1.2765e-02, 9.1548e-03],
    [2.1024e-02, -5.2973e-03, 3.4373e-03],
    [-8.8896e-03, -1.9703e-02, -1.8761e-02],
    [-1.3160e-02, -1.0523e-02, 1.9709e-03],
    [-1.5152e-03, -6.9891e-03, -7.5810e-03],
    [-1.7247e-03, 4.6560e-04, -3.3839e-03],
    [1.3617e-02, 4.7077e-03, -2.0045e-03],
    [1.0256e-02, 7.7318e-03, 1.3948e-02],
    [-1.6108e-02, -6.2151e-03, 1.1561e-03],
    [7.3407e-03, 1.5628e-02, 4.4865e-04],
    [9.5357e-04, -2.9518e-03, -1.4760e-02],
    [1.9143e-02, 1.0868e-02, 1.2264e-02],
    [4.4575e-03, 3.6682e-05, -6.8508e-03],
    [-4.5681e-04, 3.2570e-03, 7.7929e-03],
    [3.3902e-02, 3.3405e-02, 3.7454e-02],
    [-2.3001e-02, -2.4877e-03, -3.1033e-03],
    [5.0265e-02, 3.8841e-02, 3.3539e-02],
    [-4.1018e-03, -1.1095e-03, 1.5859e-03],
    [-1.2689e-01, -1.3107e-01, -2.1005e-01],
    [2.6276e-02, 1.4189e-02, -3.5963e-03],
    [-4.8679e-03, 8.8486e-03, 7.8029e-03],
    [-1.6610e-03, -4.8597e-03, -5.2060e-03],
    [-2.1010e-03, 2.3610e-03, 9.3796e-03],
    [-2.2482e-02, -2.1305e-02, -1.5087e-02],
    [-1.5753e-02, -1.0646e-02, -6.5083e-03],
    [-4.6975e-03, 5.0288e-03, -6.7390e-03],
    [1.1951e-02, 2.0712e-02, 1.6191e-02],
    [-6.3704e-03, -8.4827e-03, -9.5483e-03],
    [7.2610e-03, -9.9326e-03, -2.2978e-02],
    [-9.1904e-04, 6.2882e-03, 9.5720e-03],
    [-3.7178e-02, -3.7123e-02, -5.6713e-02],
    [-1.3373e-01, -1.0720e-01, -5.3801e-02],
    [-5.3702e-03, 8.1256e-03, 8.8397e-03],
    [-1.5247e-01, -2.1437e-01, -2.1843e-01],
    [3.1441e-02, 7.0335e-03, -9.7541e-03],
    [2.1528e-03, -8.9817e-03, -2.1023e-02],
    [3.8461e-03, -5.8957e-03, -1.5014e-02],
    [-4.3470e-03, -1.2940e-02, -1.5972e-02],
    [-5.4781e-03, -1.0842e-02, -3.0204e-03],
    [-6.5347e-03, 3.0806e-03, -1.0163e-02],
    [-5.0414e-03, -7.1503e-03, -8.9686e-04],
    [-8.5851e-03, -2.4351e-03, 1.0674e-03],
    [-9.0016e-03, -9.6493e-03, 1.5692e-03],
    [5.0914e-03, 1.2099e-02, 1.9968e-02],
    [1.3758e-02, 1.1669e-02, 8.1958e-03],
    [-1.0518e-02, -1.1575e-02, -4.1307e-03],
    [-2.8410e-02, -3.1266e-02, -2.2149e-02],
    [2.9336e-03, 3.6511e-02, 1.8717e-02],
    [-1.6703e-02, -1.6696e-02, -4.4529e-03],
    [4.8818e-02, 4.0063e-02, 8.7410e-03],
    [-1.5066e-02, -5.7328e-04, 2.9785e-03],
    [-1.7613e-02, -8.1034e-03, 1.3086e-02],
    [-9.2633e-03, 1.0803e-02, -6.3489e-03],
    [3.0851e-03, 4.7750e-04, 1.2347e-02],
    [-2.2785e-02, -2.3043e-02, -2.6005e-02],
    [-2.4787e-02, -1.5389e-02, -2.2104e-02],
    [-2.3572e-02, 1.0544e-03, 1.2361e-02],
    [-7.8915e-03, -1.2271e-03, -6.0968e-03],
    [-1.1478e-02, -1.2543e-03, 6.2679e-03],
    [-5.4229e-02, 2.6644e-02, 6.3394e-03],
    [4.4216e-03, -7.3338e-03, -1.0464e-02],
    [-4.5013e-03, 1.6082e-03, 1.4420e-02],
    [1.3673e-02, 8.8877e-03, 4.1253e-03],
    [-1.0145e-02, 9.0072e-03, 1.5695e-02],
    [-5.6234e-03, 1.1847e-03, 8.1261e-03],
    [-3.7171e-03, -5.3538e-03, 1.2590e-03],
    [2.9476e-02, 2.1424e-02, 3.0424e-02],
    [-3.4925e-02, -2.4340e-02, -2.5316e-02],
    [-3.4127e-02, -2.2406e-02, -1.0589e-02],
    [-1.7342e-02, -1.3249e-02, -1.0719e-02],
    [-2.1478e-03, -8.6051e-03, -2.9878e-03],
    [1.2089e-03, -4.2391e-03, -6.8569e-03],
    [9.0411e-04, -6.6886e-03, -6.7547e-05],
    [1.6048e-02, -1.0057e-02, -2.8929e-02],
    [1.2290e-03, 1.0163e-02, 1.8861e-02],
    [1.7264e-02, 2.7257e-04, 1.3785e-02],
    [-1.3482e-02, -3.6427e-03, 6.7481e-04],
    [4.6782e-03, -5.2423e-03, 2.4467e-03],
    [-5.9113e-03, -6.2244e-03, -1.8162e-03],
    [1.5496e-02, 1.4582e-02, 1.9514e-03],
    [7.4958e-03, 1.5886e-03, -8.2305e-03],
    [1.9086e-02, 1.6360e-03, -3.9674e-03],
    [-5.7021e-03, -2.7307e-03, -4.1066e-03],
    [1.7450e-03, 1.4602e-02, 2.5794e-02],
    [-8.2788e-04, 2.2902e-03, 4.5161e-03],
    [1.1632e-02, 8.9193e-03, -7.2813e-03],
    [7.5721e-03, 2.6784e-03, 1.1393e-02],
    [5.1939e-03, 3.6903e-03, 1.4049e-02],
    [-1.8383e-02, -2.2529e-02, -2.4477e-02],
    [5.8842e-04, -5.7874e-03, -1.4770e-02],
    [-1.6125e-02, -8.6101e-03, -1.4533e-02],
    [2.0540e-02, 2.0729e-02, 6.4338e-03],
    [3.3587e-03, -1.1226e-02, -1.6444e-02],
    [-1.4742e-03, -1.0489e-02, 1.7097e-03],
    [2.8130e-02, 2.3546e-02, 3.2791e-02],
    [-1.8532e-02, -1.2842e-02, -8.7756e-03],
    [-8.0533e-03, -1.0771e-02, -1.7536e-02],
    [-3.9009e-03, 1.6150e-02, 3.3359e-02],
    [-7.4554e-03, -1.4154e-02, -6.1910e-03],
    [3.4734e-03, -1.1370e-02, -1.0581e-02],
    [1.1476e-02, 3.9281e-03, 2.8231e-03],
    [7.1639e-03, -1.4741e-03, -3.8066e-03],
    [2.2250e-03, -8.7552e-03, -9.5719e-03],
    [2.4146e-02, 2.1696e-02, 2.8056e-02],
    [-5.4365e-03, -2.4291e-02, -1.7802e-02],
    [7.4263e-03, 1.0510e-02, 1.2705e-02],
    [6.2669e-03, 6.2658e-03, 1.9211e-02],
    [1.6378e-02, 9.4933e-03, 6.6971e-03],
    [1.7173e-02, 2.3601e-02, 2.3296e-02],
    [-1.4568e-02, -9.8279e-03, -1.1556e-02],
    [1.4431e-02, 1.4430e-02, 6.6362e-03],
    [-6.8230e-03, 1.8863e-02, 1.4555e-02],
    [6.1156e-03, 3.4700e-03, -2.6662e-03],
    [-2.6983e-03, -5.9402e-03, -9.2276e-03],
    [1.0235e-02, 7.4173e-03, -7.6243e-03],
    [-1.3255e-02, 1.9322e-02, -9.2153e-04],
    [2.4222e-03, -4.8039e-03, -1.5759e-02],
    [2.6244e-02, 2.5951e-02, 2.0249e-02],
    [1.5711e-02, 1.8498e-02, 2.7407e-03],
    [-2.1714e-03, 4.7214e-03, -2.2443e-02],
    [-7.4747e-03, 7.4166e-03, 1.4430e-02],
    [-8.3906e-03, -7.9776e-03, 9.7927e-03],
    [3.8321e-02, 9.6622e-03, -1.9268e-02],
    [-1.4605e-02, -6.7032e-03, 3.9675e-03],
]
LATENT_RGB_FACTORS_BIAS_LTXV = [-0.0571, -0.1657, -0.2512]


# =============================================================================
# Helper Functions
# =============================================================================

def build_anchor_image_tuples(
    anchor_image: str | None,
    anchor_interval: int | None,
    anchor_strength: float,
    num_frames: int,
    images: list[tuple[str, int, float]],
) -> list[tuple[str, int, float]]:
    """
    Build anchor conditioning tuples for guiding latent injection.

    Args:
        anchor_image: Explicit anchor image path, or None to use first --image
        anchor_interval: Frame interval for anchor injection
        anchor_strength: Conditioning strength for anchors
        num_frames: Total number of frames in the video
        images: Existing image conditionings (to extract first image if needed)

    Returns:
        List of (image_path, frame_idx, strength) tuples for anchor conditioning
    """
    if anchor_interval is None:
        return []

    # Determine anchor path
    if anchor_image is not None:
        anchor_path = anchor_image
    elif images:
        anchor_path = images[0][0]  # Use first i2v image
    else:
        raise ValueError("--anchor-interval requires --anchor-image or at least one --image")

    # Generate frames: [interval, 2*interval, ...] (skip 0, that's handled by i2v)
    anchor_frames = list(range(anchor_interval, num_frames, anchor_interval))
    return [(anchor_path, frame_idx, anchor_strength) for frame_idx in anchor_frames]


def build_stg_perturbation_config(
    stg_scale: float,
    stg_blocks: list[int],
    stg_mode: str,
) -> BatchedPerturbationConfig | None:
    """
    Build the perturbation config for STG based on the stg_mode.

    Args:
        stg_scale: STG guidance scale. If 0.0, returns None (STG disabled).
        stg_blocks: List of transformer block indices to perturb.
        stg_mode: Either "stg_av" (perturb both audio and video) or "stg_v" (video only).

    Returns:
        BatchedPerturbationConfig or None if STG is disabled.
    """
    if stg_scale == 0.0:
        return None

    # Always skip video self-attention for STG
    perturbations: list[Perturbation] = [
        Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=stg_blocks)
    ]

    # Optionally also skip audio self-attention (stg_av mode)
    if stg_mode == "stg_av":
        perturbations.append(
            Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=stg_blocks)
        )

    perturbation_config = PerturbationConfig(perturbations=perturbations)
    return BatchedPerturbationConfig(perturbations=[perturbation_config])


def cfg_stg_denoising_func(
    cfg_guider: CFGGuider,
    stg_guider: STGGuider,
    stg_perturbation_config: BatchedPerturbationConfig | None,
    v_context_p: torch.Tensor,
    v_context_n: torch.Tensor | None,
    a_context_p: torch.Tensor,
    a_context_n: torch.Tensor | None,
    transformer,
):
    """
    Create a denoising function that applies both CFG and STG guidance.

    This is the combined guidance function that the source LTX-2 repository uses.
    It performs up to 3 forward passes per step:
    1. Positive pass (always)
    2. Negative pass (if CFG enabled)
    3. Perturbed pass (if STG enabled)

    Args:
        cfg_guider: CFGGuider instance for classifier-free guidance.
        stg_guider: STGGuider instance for spatio-temporal guidance.
        stg_perturbation_config: Perturbation config for STG (None if disabled).
        v_context_p: Positive video context embeddings.
        v_context_n: Negative video context embeddings (None if no CFG).
        a_context_p: Positive audio context embeddings.
        a_context_n: Negative audio context embeddings (None if no CFG).
        transformer: The X0Model transformer.

    Returns:
        A denoising function compatible with euler_denoising_loop.
    """
    from ltx_core.model.transformer.modality import Modality

    def modality_from_latent_state(state: LatentState, context: torch.Tensor, sigma: float, clone_latent: bool = False) -> Modality:
        """Create a Modality object from a LatentState.

        Matches official LTX-2 code (validation_sampler.py:504) - no timestep clamping.
        Reference frames with denoise_mask=0 get timestep=0, which correctly signals
        to the attention mechanism that these are clean conditioning tokens.

        Args:
            clone_latent: If True, clone the latent tensor. Needed for multi-pass
                denoising with block swap to avoid tensor corruption.
        """
        timesteps = sigma * state.denoise_mask
        latent = state.latent.clone() if clone_latent else state.latent

        return Modality(
            enabled=True,
            latent=latent,
            timesteps=timesteps,
            positions=state.positions,
            context=context,
            context_mask=None,
        )

    def cfg_stg_denoising_step(
        video_state: LatentState,
        audio_state: LatentState,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = sigmas[step_index]

        # Detect plain block swap (without activation offload) - need to clone latents
        # to avoid tensor corruption across multiple forward passes
        plain_block_swap = (
            getattr(transformer, '_block_swap_offloader', None) is not None and
            not hasattr(transformer, '_activation_offload_verbose')
        )
        # Also check the velocity_model for X0Model wrapper
        if not plain_block_swap and hasattr(transformer, 'velocity_model'):
            vm = transformer.velocity_model
            plain_block_swap = (
                getattr(vm, '_block_swap_offloader', None) is not None and
                not hasattr(vm, '_activation_offload_verbose')
            )

        # Create positive modalities (first pass - no clone needed)
        pos_video = modality_from_latent_state(video_state, v_context_p, sigma)
        pos_audio = modality_from_latent_state(audio_state, a_context_p, sigma)

        # Forward pass with positive conditioning - SAVE original outputs
        # These are used as the baseline for all delta calculations (matching source)
        pos_denoised_video, pos_denoised_audio = transformer(
            video=pos_video, audio=pos_audio, perturbations=None
        )
        denoised_video = pos_denoised_video
        denoised_audio = pos_denoised_audio

        # Apply CFG if enabled - MASK delta to generated regions only
        # This prevents artifacts from preserved frames (timesteps=0) bleeding into generation
        if cfg_guider.enabled() and v_context_n is not None:
            # Clone latent for second pass with block swap to avoid tensor corruption
            neg_video = modality_from_latent_state(video_state, v_context_n, sigma, clone_latent=plain_block_swap)
            neg_audio = modality_from_latent_state(audio_state, a_context_n, sigma, clone_latent=plain_block_swap)

            neg_denoised_video, neg_denoised_audio = transformer(
                video=neg_video, audio=neg_audio, perturbations=None
            )

            # Calculate preserved frame ratio for CFG scale adjustment
            preserved_ratio = (1.0 - video_state.denoise_mask).mean().item()

            # Reduce effective CFG scale when many frames are preserved (>30%)
            # This prevents CFG amplification from causing blur in extension mode
            if preserved_ratio > 0.3:
                # Scale down CFG proportionally: 50% preserved -> 50% of original CFG boost
                effective_scale = 1.0 + (cfg_guider.scale - 1.0) * (1.0 - preserved_ratio)
                if step_index == 0:  # Only log once at first step
                    print(f"[AV Extension] Reducing CFG from {cfg_guider.scale:.1f} to {effective_scale:.2f} (preserved: {preserved_ratio:.1%})")
                from ltx_core.components.guiders import CFGGuider
                effective_guider = CFGGuider(effective_scale)
                cfg_delta_video = effective_guider.delta(pos_denoised_video, neg_denoised_video)
                cfg_delta_audio = effective_guider.delta(pos_denoised_audio, neg_denoised_audio)
            else:
                cfg_delta_video = cfg_guider.delta(pos_denoised_video, neg_denoised_video)
                cfg_delta_audio = cfg_guider.delta(pos_denoised_audio, neg_denoised_audio)

            # Apply CFG delta WITHOUT masking (matching source LTX-2 behavior)
            # post_process_latent will blend with clean_latent based on mask
            denoised_video = denoised_video + cfg_delta_video
            denoised_audio = denoised_audio + cfg_delta_audio

        # Apply STG if enabled - also without masking to match source
        if stg_guider.enabled() and stg_perturbation_config is not None:
            # Clone latent for third pass with block swap to avoid tensor corruption
            stg_video = modality_from_latent_state(video_state, v_context_p, sigma, clone_latent=plain_block_swap)
            stg_audio = modality_from_latent_state(audio_state, a_context_p, sigma, clone_latent=plain_block_swap)
            perturbed_video, perturbed_audio = transformer(
                video=stg_video, audio=stg_audio, perturbations=stg_perturbation_config
            )

            stg_delta_video = stg_guider.delta(pos_denoised_video, perturbed_video)
            denoised_video = denoised_video + stg_delta_video
            if perturbed_audio is not None:
                stg_delta_audio = stg_guider.delta(pos_denoised_audio, perturbed_audio)
                denoised_audio = denoised_audio + stg_delta_audio

        return denoised_video, denoised_audio

    return cfg_stg_denoising_step


class VideoConditionByMotionLatent:
    """
    SVI Pro motion conditioning using keyframe-style token appending.

    Adapts H1111's motion latent conditioning to LTX's keyframe system.
    Instead of concatenating motion latents directly to the latent tensor,
    this creates VideoConditionByKeyframeIndex items that append tokens
    with appropriate temporal position offsets.

    Motion latents guide temporal consistency by appearing at the END
    of the frame sequence, providing "future motion context" to the denoiser.
    """

    def __init__(
        self,
        motion_latent: torch.Tensor,  # Shape: [1, C, num_motion_latent, H, W]
        strength: float = 0.7,
        frame_offset: int = 0,
    ):
        """
        Args:
            motion_latent: Latent tensor from previous clip [1, C, N, H, W]
            strength: Conditioning strength (0=ignore, 1=fully condition)
            frame_offset: Additional offset for frame positioning
        """
        self.motion_latent = motion_latent
        self.strength = strength
        self.frame_offset = frame_offset

    def to_keyframe_conditionings(
        self,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list:
        """
        Convert motion latent to list of VideoConditionByKeyframeIndex items.

        Each latent frame becomes a separate keyframe conditioning at the
        appropriate temporal position (at END of sequence for motion context).

        Args:
            num_frames: Total pixel frames in the video
            dtype: Target dtype for conditioning tensors
            device: Target device

        Returns:
            List of VideoConditionByKeyframeIndex items
        """
        from ltx_core.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex

        conditionings = []
        motion_latent = self.motion_latent.to(device=device, dtype=dtype)
        num_motion_frames = motion_latent.shape[2]

        # Position motion frames at the END of the temporal sequence
        # This provides "where we're going" context for motion continuity
        for i in range(num_motion_frames):
            # Extract single latent frame: [1, C, 1, H, W]
            frame_latent = motion_latent[:, :, i:i+1, :, :]

            # Frame index at end: num_frames - num_motion_frames + i
            # Convert to pixel frame index for VideoConditionByKeyframeIndex
            frame_idx = num_frames - num_motion_frames + i + self.frame_offset

            conditionings.append(
                VideoConditionByKeyframeIndex(
                    keyframes=frame_latent,
                    frame_idx=frame_idx,
                    strength=self.strength,
                )
            )

        return conditionings


def _encode_frames_to_latent(
    frames: torch.Tensor,
    video_encoder,
    device: torch.device,
    dtype: torch.dtype,
    target_latent_frames: int,
) -> torch.Tensor:
    """
    Encode pixel frames to latent representation for motion conditioning.

    Args:
        frames: Video frames [F, H, W, C] in 0-255 uint8 or [0,1] float
        video_encoder: LTX video encoder instance
        device: Target device
        dtype: Target dtype
        target_latent_frames: Number of latent frames to return

    Returns:
        Latent tensor [1, C, target_latent_frames, H, W]
    """
    # Handle different input formats
    if frames.dtype == torch.uint8:
        frames = frames.float() / 255.0
    elif frames.max() > 1.0:
        frames = frames / 255.0

    # Convert from [F, H, W, C] to [1, C, F, H, W]
    frames = frames.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, F, H, W]
    frames = frames.to(device=device, dtype=dtype)

    # Normalize to [-1, 1] for VAE
    frames = frames * 2.0 - 1.0

    # Encode
    with torch.no_grad():
        latent = video_encoder(frames)  # [1, latent_C, lat_F, lat_H, lat_W]

    # Return last N latent frames
    return latent[:, :, -target_latent_frames:, :, :]


# =============================================================================
# AV Extension Helper Functions (Time-Based Audio-Video Masking)
# =============================================================================

AUDIO_LATENT_DOWNSAMPLE_FACTOR = 8

def time_to_video_latent_idx(
    time_sec: float,
    video_fps: float,
    time_scale_factor: int,
    video_latent_frame_count: int,
    is_end_index: bool = False,
) -> int:
    import numpy as np
    video_pixel_frame_count = (video_latent_frame_count - 1) * time_scale_factor + 1
    xp = np.array(
        [0] + list(range(1, video_pixel_frame_count + time_scale_factor, time_scale_factor))
    )
    video_pixel_frame = int(round(time_sec * video_fps))
    if is_end_index:
        latent_idx = np.searchsorted(xp, video_pixel_frame, side="right") - 1
    else:
        latent_idx = np.searchsorted(xp, video_pixel_frame, side="left")
    return max(0, min(latent_idx, video_latent_frame_count))


def time_to_audio_latent_idx(
    time_sec: float,
    sampling_rate: int,
    mel_hop_length: int,
    audio_latent_frame_count: int,
) -> int:
    """
    Convert time in seconds to audio latent frame index.

    Args:
        time_sec: Time in seconds
        sampling_rate: Audio sampling rate (e.g., 24000)
        mel_hop_length: Mel spectrogram hop length
        audio_latent_frame_count: Total number of audio latent frames

    Returns:
        Audio latent frame index (clamped to valid range)
    """
    audio_latents_per_second = sampling_rate / mel_hop_length / AUDIO_LATENT_DOWNSAMPLE_FACTOR
    audio_idx = int(round(time_sec * audio_latents_per_second))
    return max(0, min(audio_idx, audio_latent_frame_count))


# =============================================================================
# V2V Join Helper Functions
# =============================================================================

def variance_of_laplacian(image):
    """Calculate image sharpness using Laplacian variance."""
    import cv2
    return cv2.Laplacian(image, cv2.CV_64F).var()


def extract_best_transition_frame_from_end(video_path: str, frames_to_check: int = 30) -> int:
    """
    Extract the sharpest frame from the last N frames for smooth transition.

    Args:
        video_path: Path to video file
        frames_to_check: Number of frames from end to analyze

    Returns:
        Frame index (0-based) of the sharpest frame, or -1 on error
    """
    import cv2

    print(f">>> Finding best transition frame from last {frames_to_check} frames of video1...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(">>> ERROR: Failed to open video file")
        return -1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, total_frames - frames_to_check)

    best_frame_idx = total_frames - 1  # Default to last frame
    max_sharpness = -1

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for frame_idx in range(start_frame, total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate sharpness on grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = variance_of_laplacian(gray)

        if sharpness > max_sharpness:
            max_sharpness = sharpness
            best_frame_idx = frame_idx

    cap.release()

    print(f">>> Best transition frame from end: {best_frame_idx} (sharpness: {max_sharpness:.2f})")
    return best_frame_idx


def extract_best_transition_frame_from_start(video_path: str, frames_to_check: int = 30) -> int:
    """
    Extract the sharpest frame from the first N frames for smooth transition.

    Args:
        video_path: Path to video file
        frames_to_check: Number of frames from start to analyze

    Returns:
        Frame index (0-based) of the sharpest frame
    """
    import cv2

    print(f">>> Finding best transition frame from first {frames_to_check} frames of video2...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(">>> ERROR: Failed to open video file")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = min(total_frames, frames_to_check)

    best_frame_idx = 0
    max_sharpness = -1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for frame_idx in range(end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate sharpness on grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = variance_of_laplacian(gray)

        if sharpness > max_sharpness:
            max_sharpness = sharpness
            best_frame_idx = frame_idx

    cap.release()

    print(f">>> Best transition frame from start: {best_frame_idx} (sharpness: {max_sharpness:.2f})")
    return best_frame_idx


def load_video_segment(
    video_path: str,
    start_frame: int,
    end_frame: int,
    width: int,
    height: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, float]:
    """
    Load a segment of video frames and resize to target dimensions.

    Args:
        video_path: Path to video file
        start_frame: Starting frame index (inclusive)
        end_frame: Ending frame index (exclusive)
        width: Target width
        height: Target height
        device: Target device
        dtype: Target dtype

    Returns:
        Tuple of (frames tensor [F, H, W, C] normalized 0-1, fps)
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames loaded from {video_path}")

    # Convert to tensor [F, H, W, C] normalized to 0-1
    frames_tensor = torch.from_numpy(np.stack(frames)).float() / 255.0
    frames_tensor = frames_tensor.to(device=device, dtype=dtype)

    return frames_tensor, fps


def create_av_noise_mask(
    video_latent: torch.Tensor,
    audio_latent: torch.Tensor | None,
    start_time: float,
    end_time: float,
    video_fps: float,
    time_scale_factor: int,
    sampling_rate: int | None = None,
    mel_hop_length: int | None = None,
    init_video_mask: float = 0.0,
    init_audio_mask: float = 0.0,
    slope_len: int = 3,
    audio_latents_per_second: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, C, F, H, W = video_latent.shape
    video_latent_frame_count = F
    video_mask = torch.full(
        (B, 1, F, H, W),
        fill_value=init_video_mask,
        device=video_latent.device,
        dtype=torch.float32,
    )

    video_start_idx = time_to_video_latent_idx(
        start_time, video_fps, time_scale_factor, video_latent_frame_count,
        is_end_index=False,
    )
    video_end_idx = time_to_video_latent_idx(
        end_time, video_fps, time_scale_factor, video_latent_frame_count,
        is_end_index=True,
    )

    video_mask[:, :, video_start_idx:video_end_idx, :, :] = 1.0

    print(f"[AV Extension] Video mask: preserve frames 0-{video_start_idx}, "
          f"generate frames {video_start_idx}-{video_end_idx} "
          f"(total {video_latent_frame_count} latent frames)")

    audio_mask = None
    if audio_latent is not None:
        B_a, C_a, F_a, mel_bins = audio_latent.shape
        audio_latent_frame_count = F_a
        audio_mask = torch.full(
            (B_a, 1, F_a, 1),
            fill_value=init_audio_mask,
            device=audio_latent.device,
            dtype=torch.float32,
        )

        if audio_latents_per_second is not None:
            actual_rate = audio_latents_per_second
        elif sampling_rate is not None and mel_hop_length is not None:
            actual_rate = sampling_rate / mel_hop_length / AUDIO_LATENT_DOWNSAMPLE_FACTOR
            print(f"[WARNING] Using fallback audio rate calculation - may be incorrect!")
        else:
            raise ValueError("Either audio_latents_per_second or (sampling_rate, mel_hop_length) required")

        audio_start_idx = int(round(start_time * actual_rate))
        audio_end_idx = int(round(end_time * actual_rate)) + 1

        audio_start_idx = max(0, audio_start_idx)
        audio_end_idx = min(audio_end_idx, audio_latent_frame_count)

        audio_mask[:, :, audio_start_idx:audio_end_idx] = 1.0

        print(f"[AV Extension] Audio mask: preserve frames 0-{audio_start_idx}, "
              f"generate frames {audio_start_idx}-{audio_end_idx} "
              f"(total {audio_latent_frame_count} latent frames, rate={actual_rate:.2f} fps)")

    return video_mask, audio_mask


def get_video_latent_blend_coefficients(
    start_idx: int,
    end_idx: int,
    total_frames: int,
    slope_len: int = 3,
) -> tuple[list[float], list[float]]:
    """
    Calculate blend coefficients for smooth transitions at mask boundaries.

    Creates linear ramps at the start and end of the masked region
    for post-processing blending.

    Args:
        start_idx: Starting latent frame index
        end_idx: Ending latent frame index
        total_frames: Total number of latent frames
        slope_len: Length of the transition ramp

    Returns:
        Tuple of (latent_coefficients, pixel_coefficients) lists
    """
    latent_coeffs = [0.0] * total_frames

    # Ramp up at start
    for i in range(slope_len):
        idx = start_idx + i
        if idx < total_frames:
            latent_coeffs[idx] = (i + 1) / (slope_len + 1)

    # Full strength in middle
    for i in range(start_idx + slope_len, end_idx - slope_len):
        if 0 <= i < total_frames:
            latent_coeffs[i] = 1.0

    # Ramp down at end
    for i in range(slope_len):
        idx = end_idx - slope_len + i
        if 0 <= idx < total_frames:
            latent_coeffs[idx] = 1.0 - (i + 1) / (slope_len + 1)

    # Pixel coefficients are similar but for pixel-space operations
    # For simplicity, we use the same coefficients
    pixel_coeffs = latent_coeffs.copy()

    return latent_coeffs, pixel_coeffs


# =============================================================================
# Depth Control (IC-LoRA Support)
# =============================================================================

class DepthEstimator:
    """
    Depth estimation using ZoeDepth for IC-LoRA depth control.

    Provides lazy-loading of the depth model to save memory when not needed.
    Uses Intel/zoedepth-nyu-kitti for monocular depth estimation.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.model = None

    def load(self):
        """Lazy-load ZoeDepth model from HuggingFace."""
        if self.model is None:
            print(">>> Loading ZoeDepth model for depth estimation...")
            try:
                from transformers import pipeline
                self.model = pipeline(
                    "depth-estimation",
                    model="Intel/zoedepth-nyu-kitti",
                    device=0 if self.device.type == "cuda" else -1,
                )
                print(">>> ZoeDepth model loaded successfully")
            except ImportError:
                raise ImportError(
                    "ZoeDepth requires transformers>=4.35.0. "
                    "Install with: pip install transformers>=4.35.0"
                )

    def estimate_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Estimate depth from a single image tensor.

        Args:
            image: Image tensor [H, W, C] in range [0, 1] or [0, 255]

        Returns:
            Depth tensor [H, W] normalized to [0, 1]
        """
        from PIL import Image as PILImage
        import numpy as np

        self.load()

        # Convert tensor to PIL Image
        if image.max() <= 1.0:
            image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        else:
            image_np = image.cpu().numpy().astype(np.uint8)

        pil_image = PILImage.fromarray(image_np)

        # Run depth estimation
        result = self.model(pil_image)
        depth = result["depth"]

        # Convert PIL depth to tensor and normalize
        depth_np = np.array(depth).astype(np.float32)
        depth_min, depth_max = depth_np.min(), depth_np.max()
        if depth_max > depth_min:
            depth_np = (depth_np - depth_min) / (depth_max - depth_min)
        else:
            depth_np = np.zeros_like(depth_np)

        return torch.from_numpy(depth_np)

    def estimate_video(
        self,
        frames: torch.Tensor,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> torch.Tensor:
        """
        Estimate depth for each frame in a video.

        Args:
            frames: Video tensor [1, C, F, H, W] in range [-1, 1]
            progress_callback: Optional callback(current, total) for progress

        Returns:
            Depth tensor [1, 3, F, H, W] in range [0, 1] (RGB, same value in all channels)
        """
        self.load()

        # Convert from [1, C, F, H, W] to list of [H, W, C] frames
        frames = frames.squeeze(0)  # [C, F, H, W]
        frames = (frames + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        frames = frames.permute(1, 2, 3, 0)  # [F, H, W, C]

        num_frames = frames.shape[0]
        depth_frames = []

        for i in range(num_frames):
            frame = frames[i]  # [H, W, C]
            depth = self.estimate_image(frame)  # [H, W]

            # Replicate to 3 channels for VAE encoding (expects RGB)
            # Use expand() instead of repeat() - creates view without memory allocation
            depth_rgb = depth.unsqueeze(-1).expand(-1, -1, 3).contiguous()  # [H, W, 3]
            depth_frames.append(depth_rgb)

            if progress_callback:
                progress_callback(i + 1, num_frames)

        # Stack and convert back to [1, C, F, H, W]
        depth_video = torch.stack(depth_frames, dim=0)  # [F, H, W, C]
        depth_video = depth_video.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, F, H, W]

        return depth_video

    def unload(self):
        """Free model memory."""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            print(">>> ZoeDepth model unloaded")


def load_or_estimate_depth(
    depth_video: str | None,
    depth_image: str | None,
    estimate_depth: bool,
    source_video: str | None,
    source_image: str | None,
    num_frames: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """
    Load pre-generated depth maps OR estimate depth from source.

    Priority:
    1. depth_video - load pre-generated depth video
    2. depth_image - load single depth image and repeat for all frames
    3. estimate_depth - estimate from source_video or source_image

    Args:
        depth_video: Path to pre-generated depth map video
        depth_image: Path to single depth map image
        estimate_depth: Whether to auto-estimate depth
        source_video: Path to source video for depth estimation
        source_image: Path to source image for depth estimation
        num_frames: Number of frames to generate
        height: Output height (pixel space)
        width: Output width (pixel space)
        device: Target device
        dtype: Target dtype

    Returns:
        Depth tensor [1, 3, F, H, W] normalized to [0, 1], or None if no depth source
    """
    import torch.nn.functional as F

    if depth_video:
        # Load pre-generated depth video
        print(f">>> Loading depth video: {depth_video}")
        depth_tensor = load_video_conditioning(
            video_path=depth_video,
            height=height,
            width=width,
            frame_cap=num_frames,
            dtype=dtype,
            device=torch.device("cpu"),
        )
        # Normalize to [0, 1] if needed (load_video_conditioning returns [-1, 1])
        depth_tensor = (depth_tensor + 1.0) / 2.0
        return depth_tensor

    elif depth_image:
        # Load single depth image and repeat for all frames
        print(f">>> Loading depth image: {depth_image}")
        from PIL import Image as PILImage
        import numpy as np

        img = PILImage.open(depth_image).convert("RGB")
        img = img.resize((width, height), PILImage.Resampling.LANCZOS)
        img_np = np.array(img).astype(np.float32) / 255.0  # [H, W, C]

        # Convert to tensor [1, C, 1, H, W]
        depth_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)

        # Expand for all frames - creates view without memory allocation
        # Note: expand() creates a view with stride 0, downstream ops handle this correctly
        depth_tensor = depth_tensor.expand(1, 1, num_frames, -1, -1).contiguous()

        return depth_tensor.to(dtype=dtype)

    elif estimate_depth:
        # Estimate depth from source video or image
        estimator = DepthEstimator(device)

        if source_video:
            print(f">>> Estimating depth from video: {source_video}")
            # Load source video
            source_tensor = load_video_conditioning(
                video_path=source_video,
                height=height,
                width=width,
                frame_cap=num_frames,
                dtype=dtype,
                device=torch.device("cpu"),
            )
            # Estimate depth for each frame
            depth_tensor = estimator.estimate_video(
                source_tensor,
                progress_callback=lambda c, t: print(f">>> Depth estimation: {c}/{t} frames", end="\r") if c % 10 == 0 or c == t else None,
            )
            print()  # Newline after progress

        elif source_image:
            print(f">>> Estimating depth from image: {source_image}")
            from PIL import Image as PILImage
            import numpy as np

            img = PILImage.open(source_image).convert("RGB")
            img = img.resize((width, height), PILImage.Resampling.LANCZOS)
            img_np = np.array(img).astype(np.float32) / 255.0

            img_tensor = torch.from_numpy(img_np)  # [H, W, C]
            depth_frame = estimator.estimate_image(img_tensor)  # [H, W]

            # Replicate to RGB and all frames
            # Use expand() instead of repeat() - creates view without memory allocation
            depth_rgb = depth_frame.unsqueeze(-1).expand(-1, -1, 3).contiguous()  # [H, W, 3]
            depth_tensor = depth_rgb.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # [1, C, 1, H, W]
            depth_tensor = depth_tensor.expand(1, 1, num_frames, -1, -1).contiguous()  # [1, C, F, H, W]

        else:
            print(">>> Warning: --estimate-depth requires --image or --input-video")
            estimator.unload()
            return None

        estimator.unload()
        return depth_tensor.to(dtype=dtype)

    return None


def encode_depth_conditioning(
    depth_tensor: torch.Tensor,
    video_encoder,
    num_frames: int,
    strength: float,
    device: torch.device,
    dtype: torch.dtype,
) -> list:
    """
    Encode depth tensor to guiding latent conditioning.

    Uses tiled encoding (same pattern as V2V) to handle long videos without OOM.

    Args:
        depth_tensor: Depth maps [1, 3, F, H, W] in range [0, 1]
        video_encoder: LTX video encoder instance
        num_frames: Total pixel frames in the video
        strength: Conditioning strength (1.0 = full conditioning)
        device: Target device
        dtype: Target dtype

    Returns:
        List of VideoConditionByKeyframeIndex for in-context conditioning
    """
    from ltx_core.conditioning import VideoConditionByKeyframeIndex

    conditionings = []

    # Convert depth to [-1, 1] range for VAE encoding
    depth_for_encoding = depth_tensor * 2.0 - 1.0

    # Tiled encoding parameters (matching V2V pattern)
    tile_size = 17  # 2 latent frames + 1
    tile_overlap = 8
    tile_stride = tile_size - tile_overlap

    actual_frames = depth_for_encoding.shape[2]
    frame_idx = 0

    while frame_idx < actual_frames:
        # Calculate chunk bounds
        chunk_end = min(frame_idx + tile_size, actual_frames)
        chunk_frames = chunk_end - frame_idx

        # Ensure valid frame count (8k+1) for VAE
        valid_frames = ((chunk_frames - 1) // 8) * 8 + 1
        if valid_frames < 9:
            break
        chunk_end = frame_idx + valid_frames

        # Extract chunk and encode
        chunk = depth_for_encoding[:, :, frame_idx:chunk_end, :, :].to(device=device, dtype=dtype)
        with torch.no_grad():
            encoded_chunk = video_encoder(chunk)
        del chunk
        torch.cuda.empty_cache()

        # Move to CPU to save memory
        encoded_chunk_cpu = encoded_chunk.cpu()
        del encoded_chunk
        torch.cuda.empty_cache()

        # Add as keyframe conditioning
        conditionings.append(
            VideoConditionByKeyframeIndex(
                keyframes=encoded_chunk_cpu,
                frame_idx=frame_idx,
                strength=strength,
            )
        )

        # Move to next tile
        frame_idx += tile_stride
        if frame_idx >= actual_frames - 8:
            break

    return conditionings


# =============================================================================
# Argument Parser
# =============================================================================

def resolve_path(path: str) -> str:
    """Resolve a path to an absolute path."""
    return str(Path(path).expanduser().resolve().as_posix())


class ImageAction(argparse.Action):
    """Parse image conditioning arguments: PATH FRAME_IDX STRENGTH"""
    def __call__(self, parser, namespace, values, option_string=None):
        path, frame_idx, strength_str = values
        resolved_path = resolve_path(path)
        frame_idx = int(frame_idx)
        strength = float(strength_str)
        current = getattr(namespace, self.dest) or []
        current.append((resolved_path, frame_idx, strength))
        setattr(namespace, self.dest, current)


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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LTX-2 Video Generation with advanced memory management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic text-to-video generation
  python ltx_generate_video.py \\
    --checkpoint-path model.safetensors \\
    --spatial-upsampler-path upsampler.safetensors \\
    --distilled-lora distilled.safetensors \\
    --prompt "A cat playing piano" \\
    --output-path output.mp4

  # Image-to-video with conditioning
  python ltx_generate_video.py \\
    --checkpoint-path model.safetensors \\
    --spatial-upsampler-path upsampler.safetensors \\
    --distilled-lora distilled.safetensors \\
    --prompt "The cat starts playing" \\
    --image input.jpg 0 0.9 \\
    --output-path output.mp4

  # Memory-optimized generation with offloading
  python ltx_generate_video.py \\
    --checkpoint-path model.safetensors \\
    --spatial-upsampler-path upsampler.safetensors \\
    --distilled-lora distilled.safetensors \\
    --prompt "A beautiful sunset" \\
    --offload \\
    --enable-fp8 \\
    --output-path output.mp4
        """
    )

    # ==========================================================================
    # Required Model Paths
    # ==========================================================================
    model_group = parser.add_argument_group("Model Paths")
    model_group.add_argument(
        "--checkpoint-path",
        type=resolve_path,
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Path to LTX-2 model checkpoint (default: {DEFAULT_CHECKPOINT_PATH}).",
    )
    model_group.add_argument(
        "--gemma-root",
        type=resolve_path,
        default=DEFAULT_GEMMA_ROOT,
        help=f"Path to Gemma text encoder directory (default: {DEFAULT_GEMMA_ROOT}).",
    )
    model_group.add_argument(
        "--spatial-upsampler-path",
        type=resolve_path,
        default=DEFAULT_SPATIAL_UPSAMPLER_PATH,
        help=f"Path to spatial upsampler model (default: {DEFAULT_SPATIAL_UPSAMPLER_PATH}).",
    )
    model_group.add_argument(
        "--distilled-lora",
        dest="distilled_lora",
        action=LoraAction,
        nargs="+",
        metavar=("PATH", "STRENGTH"),
        default=None,
        help=f"Distilled LoRA for stage 2 (default: {DEFAULT_DISTILLED_LORA_PATH} 1.0).",
    )
    model_group.add_argument(
        "--distilled-checkpoint",
        dest="distilled_checkpoint",
        action="store_true",
        default=False,
        help="Use distilled model settings for stage 1. "
             "Use this when the main checkpoint is a distilled model (not requiring CFG).",
    )
    model_group.add_argument(
        "--stage2-checkpoint",
        type=resolve_path,
        default=None,
        help="Path to a separate checkpoint for stage 2 refinement (full model, not LoRA). "
             "If not specified, uses the main checkpoint with --distilled-lora applied.",
    )
    model_group.add_argument(
        "--stage2-steps",
        type=int,
        default=3,
        help="Number of denoising steps for stage 2 refinement (default: 3). "
             "Uses LTX2Scheduler to generate sigma schedule.",
    )
    model_group.add_argument(
        "--vae",
        type=resolve_path,
        default=None,
        help="Path to separate VAE weights file (e.g., diffusion_pytorch_model_vae.safetensors). "
             "If not specified, VAE weights are loaded from the main checkpoint.",
    )

    # ==========================================================================
    # Generation Parameters
    # ==========================================================================
    gen_group = parser.add_argument_group("Generation Parameters")
    gen_group.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt describing the desired video content.",
    )
    gen_group.add_argument(
        "--negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt for unwanted content (default: comprehensive quality prompt).",
    )
    gen_group.add_argument(
        "--output-path",
        type=resolve_path,
        required=True,
        help="Output video file path (.mp4).",
    )
    gen_group.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED}).",
    )
    gen_group.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Video width in pixels, must be divisible by 32 (default: {DEFAULT_WIDTH}).",
    )
    gen_group.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Video height in pixels, must be divisible by 32 (default: {DEFAULT_HEIGHT}).",
    )
    gen_group.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help=f"Number of frames (8*K + 1, e.g., 121 = 5s at 24fps) (default: {DEFAULT_NUM_FRAMES}).",
    )
    gen_group.add_argument(
        "--frame-rate",
        type=float,
        default=DEFAULT_FRAME_RATE,
        help=f"Video frame rate in fps (default: {DEFAULT_FRAME_RATE}).",
    )
    gen_group.add_argument(
        "--num-inference-steps",
        type=int,
        default=DEFAULT_NUM_INFERENCE_STEPS,
        help=f"Number of denoising steps for stage 1 (default: {DEFAULT_NUM_INFERENCE_STEPS}).",
    )
    gen_group.add_argument(
        "--cfg-guidance-scale",
        type=float,
        default=DEFAULT_CFG_GUIDANCE_SCALE,
        help=f"CFG guidance scale (default: {DEFAULT_CFG_GUIDANCE_SCALE}).",
    )
    gen_group.add_argument(
        "--stg-scale",
        type=float,
        default=1.0,
        help="STG (Spatio-Temporal Guidance) scale. 0.0 disables STG. "
             "Recommended: 1.0 for dev model. (default: 1.0)",
    )
    gen_group.add_argument(
        "--stg-blocks",
        type=int,
        nargs="+",
        default=[29],
        help="Transformer block indices to perturb for STG. (default: 29)",
    )
    gen_group.add_argument(
        "--stg-mode",
        type=str,
        choices=["stg_av", "stg_v"],
        default="stg_av",
        help="STG mode: 'stg_av' perturbs both audio and video self-attention, "
             "'stg_v' perturbs video only. (default: stg_av)",
    )

    # ==========================================================================
    # Image Conditioning (I2V)
    # ==========================================================================
    i2v_group = parser.add_argument_group("Image Conditioning (I2V)")
    i2v_group.add_argument(
        "--image",
        dest="images",
        action=ImageAction,
        nargs=3,
        metavar=("PATH", "FRAME_IDX", "STRENGTH"),
        default=[],
        help="Image conditioning: path, frame index, strength. Can be repeated.",
    )

    # ==========================================================================
    # Anchor Image Conditioning
    # ==========================================================================
    anchor_group = parser.add_argument_group("Anchor Image Conditioning")
    anchor_group.add_argument(
        "--anchor-image",
        type=resolve_path,
        default=None,
        help="Anchor image path for periodic guidance. If not provided but --anchor-interval is set, uses first --image.",
    )
    anchor_group.add_argument(
        "--anchor-interval",
        type=int,
        default=None,
        help="Frame interval for anchor injection (e.g., 60). Anchors injected at [interval, 2*interval, ...].",
    )
    anchor_group.add_argument(
        "--anchor-strength",
        type=float,
        default=0.8,
        help="Conditioning strength for anchor images (default: 0.8).",
    )
    anchor_group.add_argument(
        "--anchor-decay",
        type=str,
        choices=["none", "linear", "cosine", "sigmoid"],
        default="none",
        help="Decay schedule for anchor constraints. 'none' = constant strength, "
             "'cosine' = smooth decay (recommended), 'linear' = steady decay, "
             "'sigmoid' = late-stage release. Decay allows anchors to guide structure "
             "early but permit motion later. (default: none)",
    )

    # ==========================================================================
    # LoRA Support
    # ==========================================================================
    lora_group = parser.add_argument_group("LoRA Support")
    lora_group.add_argument(
        "--lora",
        dest="loras",
        action=LoraAction,
        nargs="+",
        metavar=("PATH", "STRENGTH"),
        default=[],
        help="User LoRA for stage 1 (base generation): path and optional strength. Can be repeated.",
    )
    lora_group.add_argument(
        "--stage2-lora",
        dest="stage2_loras",
        action=LoraAction,
        nargs="+",
        metavar=("PATH", "STRENGTH"),
        default=[],
        help="User LoRA for stage 2 (refinement) only: path and optional strength. Can be repeated.",
    )

    # ==========================================================================
    # Depth Control (IC-LoRA)
    # ==========================================================================
    depth_group = parser.add_argument_group("Depth Control (IC-LoRA)")
    depth_group.add_argument(
        "--depth-video",
        type=resolve_path,
        default=None,
        help="Pre-generated depth map video (grayscale or RGB depth visualization).",
    )
    depth_group.add_argument(
        "--depth-image",
        type=resolve_path,
        default=None,
        help="Depth map image to apply uniformly across all frames.",
    )
    depth_group.add_argument(
        "--estimate-depth",
        action="store_true",
        help="Auto-estimate depth from --image or --input-video using ZoeDepth.",
    )
    depth_group.add_argument(
        "--depth-strength",
        type=float,
        default=1.0,
        help="Strength of depth conditioning (default: 1.0).",
    )
    depth_group.add_argument(
        "--depth-stage2",
        action="store_true",
        help="Also apply depth conditioning to stage 2 refinement (default: stage 1 only).",
    )

    # ==========================================================================
    # Memory Optimization
    # ==========================================================================
    mem_group = parser.add_argument_group("Memory Optimization")
    mem_group.add_argument(
        "--offload",
        action="store_true",
        help="Enable CPU/GPU offloading to reduce VRAM usage.",
    )
    mem_group.add_argument(
        "--enable-fp8",
        action="store_true",
        help="Enable FP8 mode for transformer (reduces memory, calculations in bfloat16).",
    )
    # DiT (main transformer) block swapping
    mem_group.add_argument(
        "--enable-dit-block-swap",
        action="store_true",
        help="Enable block swapping for main DiT transformer (stage 1).",
    )
    mem_group.add_argument(
        "--dit-blocks-in-memory",
        type=int,
        default=22,
        help="Number of DiT transformer blocks to keep in GPU (default: 22).",
    )
    # Text encoder block swapping
    mem_group.add_argument(
        "--enable-text-encoder-block-swap",
        action="store_true",
        help="Enable block swapping for text encoder (Gemma).",
    )
    mem_group.add_argument(
        "--text-encoder-blocks-in-memory",
        type=int,
        default=6,
        help="Number of text encoder layers to keep in GPU (default: 6). "
             "The Gemma-3-12B text encoder has 48 layers.",
    )
    # Refiner block swapping
    mem_group.add_argument(
        "--enable-refiner-block-swap",
        action="store_true",
        help="Enable block swapping for refiner transformer (stage 2).",
    )
    mem_group.add_argument(
        "--refiner-blocks-in-memory",
        type=int,
        default=22,
        help="Number of refiner transformer blocks to keep in GPU (default: 22).",
    )
    # FFN chunking for long sequences
    mem_group.add_argument(
        "--ffn-chunk-size",
        type=int,
        default=None,
        help="Enable FFN chunking for long sequences (reduces peak memory). "
             "Recommended: 4096 for 1000+ frame videos. Default: None (disabled).",
    )
    # Activation offloading for extreme memory savings
    mem_group.add_argument(
        "--enable-activation-offload",
        action="store_true",
        help="Offload activations to CPU between transformer blocks. "
             "Enables processing very long videos that wouldn't fit in VRAM. "
             "Trade-off: ~10-20x slower but uses minimal GPU memory.",
    )
    # Temporal chunking for very long videos
    mem_group.add_argument(
        "--temporal-chunk-size",
        type=int,
        default=0,
        help="Process video in temporal chunks of this many tokens. "
             "0 = disabled, try 400000 for very long videos. "
             "Requires --enable-activation-offload. Maintains full attention context "
             "by streaming K/V from CPU. Much slower but uses minimal GPU memory.",
    )

    # ==========================================================================
    # Audio Control
    # ==========================================================================
    audio_group = parser.add_argument_group("Audio Control")
    audio_group.add_argument(
        "--disable-audio",
        action="store_true",
        help="Disable audio generation (video only output).",
    )
    audio_group.add_argument(
        "--audio",
        type=resolve_path,
        default=None,
        help="Path to input audio file (wav, mp3, etc.) for audio-conditioned generation.",
    )
    audio_group.add_argument(
        "--audio-strength",
        type=float,
        default=1.0,
        help="Audio conditioning strength (1.0=frozen/exact, 0.0=regenerate). Default: 1.0.",
    )

    # ==========================================================================
    # Pipeline Selection
    # ==========================================================================
    pipeline_group = parser.add_argument_group("Pipeline Selection")
    pipeline_group.add_argument(
        "--one-stage",
        action="store_true",
        help="Use one-stage pipeline (faster, generates at full resolution directly). "
             "Default is two-stage (half res + upsample + refine).",
    )
    pipeline_group.add_argument(
        "--refine-only",
        action="store_true",
        help="Use refine-only pipeline (stage 2 only on input video). "
             "Requires --input-video.",
    )

    # ==========================================================================
    # Video Input (V2V / Refine)
    # ==========================================================================
    v2v_group = parser.add_argument_group("Video Input (V2V / Refine)")
    v2v_group.add_argument(
        "--input-video",
        type=resolve_path,
        default=None,
        help="Input video path for video-to-video refinement.",
    )
    v2v_group.add_argument(
        "--refine-strength",
        type=float,
        default=0.3,
        help="Amount of noise to add before refinement (0=none, 1=full denoise). Default: 0.3",
    )
    v2v_group.add_argument(
        "--refine-steps",
        type=int,
        default=10,
        help="Number of refinement steps for refine-only mode. Default: 10",
    )

    # ==========================================================================
    # V2A Mode (Video-to-Audio)
    # ==========================================================================
    v2a_group = parser.add_argument_group("V2A Mode (Video-to-Audio)")
    v2a_group.add_argument(
        "--v2a-mode",
        action="store_true",
        help="Video-to-Audio mode: preserve input video exactly and generate new audio. "
             "Requires --input-video.",
    )

    # ==========================================================================
    # SVI Pro (Multi-Clip) Mode
    # ==========================================================================
    svi_group = parser.add_argument_group("SVI Pro (Multi-Clip) Mode")
    svi_group.add_argument(
        "--svi-mode",
        action="store_true",
        help="Enable SVI Pro mode for multi-clip generation with motion continuity.",
    )
    svi_group.add_argument(
        "--num-clips",
        type=int,
        default=2,
        help="Number of clips to generate in SVI mode (default: 2).",
    )
    svi_group.add_argument(
        "--num-motion-latent",
        type=int,
        default=2,
        help="Number of latent frames from previous clip to use for motion conditioning (default: 2).",
    )
    svi_group.add_argument(
        "--num-motion-frame",
        type=int,
        default=1,
        help="Frame offset from end of clip for next input image (1=last, 4=4th from last). Default: 1.",
    )
    svi_group.add_argument(
        "--seed-multiplier",
        type=int,
        default=42,
        help="Per-clip seed variation: seed = base_seed + clip_idx * multiplier (default: 42).",
    )
    svi_group.add_argument(
        "--overlap-frames",
        type=int,
        default=1,
        help="Number of overlapping frames between clips for smooth transitions (default: 1).",
    )
    svi_group.add_argument(
        "--extend-video",
        type=resolve_path,
        default=None,
        help="Input video path to extend using SVI Pro. Enables video extension mode.",
    )
    svi_group.add_argument(
        "--prepend-original",
        action="store_true",
        default=True,
        help="Prepend original video frames when extending (default: True).",
    )
    svi_group.add_argument(
        "--prompt-list",
        type=str,
        nargs="*",
        default=None,
        help="List of prompts for multi-clip generation. One prompt per clip. If fewer prompts than clips, last prompt is repeated.",
    )

    # ==========================================================================
    # Advanced Options
    # ==========================================================================
    adv_group = parser.add_argument_group("Advanced Options")
    adv_group.add_argument(
        "--enhance-prompt",
        action="store_true",
        help="Use Gemma to enhance the prompt before generation.",
    )
    adv_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    # ==========================================================================
    # Latent Normalization (fixes overbaking and audio clipping)
    # ==========================================================================
    norm_group = parser.add_argument_group("Latent Normalization (Quality Fix)")
    norm_group.add_argument(
        "--latent-norm",
        type=str,
        choices=["none", "stat", "adain"],
        default="none",
        help="Latent normalization mode: 'none' (disabled), 'stat' (statistical normalization), "
             "'adain' (adaptive instance normalization with reference). Default: none",
    )
    norm_group.add_argument(
        "--latent-norm-factors",
        type=str,
        default="0.9,0.75,0.5,0.25,0.0",
        help="Per-step normalization factors (comma-separated). Higher values = stronger normalization. "
             "Recommended: stronger early, weaker late. Default: '0.9,0.75,0.5,0.25,0.0'",
    )
    norm_group.add_argument(
        "--latent-norm-target-mean",
        type=float,
        default=0.0,
        help="Target mean for statistical normalization. Default: 0.0",
    )
    norm_group.add_argument(
        "--latent-norm-target-std",
        type=float,
        default=1.0,
        help="Target standard deviation for statistical normalization. Default: 1.0",
    )
    norm_group.add_argument(
        "--latent-norm-percentile",
        type=float,
        default=95.0,
        help="Percentile for outlier filtering in stat norm (50-100). Default: 95.0",
    )
    norm_group.add_argument(
        "--latent-norm-clip-outliers",
        action="store_true",
        help="Clip outliers to normalized bounds (stat norm only).",
    )
    norm_group.add_argument(
        "--latent-norm-video-only",
        action="store_true",
        help="Apply normalization to video latents only (skip audio).",
    )
    norm_group.add_argument(
        "--latent-norm-audio-only",
        action="store_true",
        help="Apply normalization to audio latents only (skip video).",
    )

    # ==========================================================================
    # Preview Generation
    # ==========================================================================
    preview_group = parser.add_argument_group("Preview Generation")
    preview_group.add_argument(
        "--preview-dir",
        type=str,
        default=None,
        help="Directory to save preview images during denoising (enables preview mode).",
    )
    preview_group.add_argument(
        "--preview-interval",
        type=int,
        default=1,
        help="Save preview every N denoising steps (default: 1).",
    )
    preview_group.add_argument(
        "--preview-max-height",
        type=int,
        default=200,
        help="Maximum height of preview images in pixels (default: 200).",
    )
    preview_group.add_argument(
        "--preview-suffix",
        type=str,
        default="",
        help="Unique suffix for preview filenames (for multi-instance support).",
    )

    # ==========================================================================
    # Video Continuation (Enhanced)
    # ==========================================================================
    cont_group = parser.add_argument_group("Video Continuation (Enhanced)")
    cont_group.add_argument(
        "--freeze-frames",
        type=int,
        default=0,
        help="Number of frames from input video to freeze during denoising (default: 0 = disabled).",
    )
    cont_group.add_argument(
        "--freeze-transition",
        type=int,
        default=4,
        help="Number of frames for gradual transition from frozen to generated (default: 4).",
    )

    # ==========================================================================
    # AV Extension Mode (Time-Based Masking)
    # ==========================================================================
    av_ext_group = parser.add_argument_group("AV Extension Mode (Time-Based Audio-Video Continuation)")
    av_ext_group.add_argument(
        "--av-extend-from",
        type=resolve_path,
        default=None,
        help="Input video path to extend/continue from using time-based AV masking.",
    )
    av_ext_group.add_argument(
        "--av-extend-start-time",
        type=float,
        default=None,
        help="Time (seconds) to start generating new content. Default: end of input video.",
    )
    av_ext_group.add_argument(
        "--av-extend-end-time",
        type=float,
        default=None,
        help="Time (seconds) to stop generation. Default: start_time + 5 seconds.",
    )
    av_ext_group.add_argument(
        "--av-extend-steps",
        type=int,
        default=8,
        help="Denoising steps for AV extension (default: 8).",
    )
    av_ext_group.add_argument(
        "--av-extend-terminal",
        type=float,
        default=0.1,
        help="Terminal sigma for partial denoising, enables smooth continuation (default: 0.1).",
    )
    av_ext_group.add_argument(
        "--av-slope-len",
        type=int,
        default=3,
        help="Transition smoothness at mask boundaries in latent frames (default: 3).",
    )
    av_ext_group.add_argument(
        "--av-no-stage2",
        action="store_true",
        help="Skip stage 2 refinement for AV extension (faster but lower quality).",
    )

    # ==========================================================================
    # V2V Join Mode
    # ==========================================================================
    v2v_join_group = parser.add_argument_group("V2V Join Mode")
    v2v_join_group.add_argument(
        "--v2v-join-video1",
        type=resolve_path,
        default=None,
        help="First video for V2V join (transition will be generated from end of this video).",
    )
    v2v_join_group.add_argument(
        "--v2v-join-video2",
        type=resolve_path,
        default=None,
        help="Second video for V2V join (transition will connect to start of this video).",
    )
    v2v_join_group.add_argument(
        "--v2v-join-frames-check1",
        type=int,
        default=30,
        help="Number of frames to check from end of video1 for sharpest transition point (default: 30).",
    )
    v2v_join_group.add_argument(
        "--v2v-join-frames-check2",
        type=int,
        default=30,
        help="Number of frames to check from start of video2 for sharpest transition point (default: 30).",
    )
    v2v_join_group.add_argument(
        "--v2v-join-preserve1",
        type=float,
        default=5.0,
        help="Seconds to preserve from end of video1 in transition (default: 5.0).",
    )
    v2v_join_group.add_argument(
        "--v2v-join-preserve2",
        type=float,
        default=5.0,
        help="Seconds to preserve from start of video2 in transition (default: 5.0).",
    )
    v2v_join_group.add_argument(
        "--v2v-join-transition-time",
        type=float,
        default=10.0,
        help="Total seconds for generated transition between preserved sections (default: 10.0).",
    )
    v2v_join_group.add_argument(
        "--v2v-join-steps",
        type=int,
        default=8,
        help="Denoising steps for V2V join transition generation (default: 8).",
    )
    v2v_join_group.add_argument(
        "--v2v-join-terminal",
        type=float,
        default=0.1,
        help="Terminal sigma for V2V join partial denoising (default: 0.1).",
    )

    # ==========================================================================
    # Sliding Window (Long Video)
    # ==========================================================================
    sliding_group = parser.add_argument_group("Sliding Window (Long Video)")
    sliding_group.add_argument(
        "--enable-sliding-window",
        action="store_true",
        help="Enable sliding window mode for long videos. Required to use sliding windows.",
    )
    sliding_group.add_argument(
        "--sliding-window-size",
        type=int,
        default=129,
        help="Frames per window, must be 8n+1 for latent alignment (default: 129).",
    )
    sliding_group.add_argument(
        "--sliding-window-overlap",
        type=int,
        default=9,
        help="Overlapping frames between windows, should be 8n+1 (default: 9).",
    )
    sliding_group.add_argument(
        "--sliding-window-overlap-noise",
        type=float,
        default=0.0,
        help="Noise level (0-100) for overlap blending to reduce seams (default: 0.0).",
    )
    sliding_group.add_argument(
        "--sliding-window-color-correction",
        type=float,
        default=0.0,
        help="LAB color correction strength (0-1) for consistent colors across windows (default: 0.0).",
    )

    return parser.parse_args()


# =============================================================================
# Memory Management Utilities
# =============================================================================

def offload_model(model: torch.nn.Module, target: str = "cpu") -> torch.nn.Module:
    """Offload a model to CPU or back to GPU."""
    if model is not None:
        model = model.to(target, non_blocking=True)
    return model


def synchronize_and_cleanup():
    """Synchronize CUDA and cleanup memory."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reconfigure_block_swap(
    transformer,
    new_blocks_in_memory: int,
    device: torch.device,
    enable_activation_offload: bool = False,
    temporal_chunk_size: int = 0,
) -> tuple:
    """
    Reconfigure block swapping with fewer blocks in GPU after an OOM.

    Args:
        transformer: The transformer model with block swap enabled
        new_blocks_in_memory: New (reduced) number of blocks to keep in GPU
        device: Target GPU device
        enable_activation_offload: If True, use activation offloading (moves activations to CPU between blocks)
        temporal_chunk_size: If > 0, process video in temporal chunks

    Returns:
        Tuple of (new_offloader, new_blocks_in_memory) or (None, 0) if at minimum
    """
    from ltx_core.model.transformer.model import X0Model

    # Get the underlying LTXModel
    if isinstance(transformer, X0Model):
        ltx_model = transformer.velocity_model
    else:
        ltx_model = transformer

    # Get current offloader
    offloader = getattr(ltx_model, "_block_swap_offloader", None)
    if offloader is None:
        print("[BlockSwap] No offloader found, cannot reconfigure")
        return None, 0

    num_blocks = len(ltx_model.transformer_blocks)

    # Ensure we have at least 1 block
    if new_blocks_in_memory < 1:
        print(f"[BlockSwap] Cannot reduce below 1 block in GPU")
        return None, 0

    print(f"[BlockSwap] Reconfiguring: {new_blocks_in_memory}/{num_blocks} blocks in GPU...")

    # Wait for any pending async operations
    for idx in range(num_blocks):
        if idx in offloader.futures:
            offloader._wait_blocks_move(idx)

    # Shutdown the ThreadPoolExecutor
    offloader.thread_pool.shutdown(wait=True)
    offloader.futures.clear()

    # Synchronize CUDA
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Move all blocks to CPU
    from ltx_pipelines.utils.custom_offloading_utils import weighs_to_device
    for block in ltx_model.transformer_blocks:
        weighs_to_device(block, "cpu")

    # Clear GPU memory
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

    # Restore original method temporarily
    if hasattr(ltx_model, "_original_process_transformer_blocks"):
        ltx_model._process_transformer_blocks = ltx_model._original_process_transformer_blocks

    # Clean up old offloader attributes
    if hasattr(ltx_model, "_block_swap_offloader"):
        del ltx_model._block_swap_offloader
    if hasattr(ltx_model, "_blocks_to_swap"):
        del ltx_model._blocks_to_swap
    if hasattr(ltx_model, "_blocks_ref"):
        del ltx_model._blocks_ref

    if isinstance(transformer, X0Model):
        if hasattr(transformer, "_block_swap_offloader"):
            del transformer._block_swap_offloader
        if hasattr(transformer, "_blocks_to_swap"):
            del transformer._blocks_to_swap
        if hasattr(transformer, "_blocks_ref"):
            del transformer._blocks_ref

    # Re-enable block swap with new configuration
    if enable_activation_offload:
        new_offloader = enable_block_swap_with_activation_offload(
            transformer,
            blocks_in_memory=new_blocks_in_memory,
            device=device,
            verbose=True,
            temporal_chunk_size=temporal_chunk_size,
        )
    else:
        new_offloader = enable_block_swap(
            transformer,
            blocks_in_memory=new_blocks_in_memory,
            device=device,
        )

    return new_offloader, new_blocks_in_memory


def set_ffn_chunk_size(transformer: torch.nn.Module, chunk_size: int | None) -> None:
    """
    Set FFN chunk size on all transformer blocks for memory-efficient inference.

    FFN chunking processes the feed-forward network in smaller chunks along the
    sequence dimension, reducing peak memory for long videos (1000+ frames).
    This is mathematically equivalent to non-chunked processing.

    Args:
        transformer: The transformer model (X0Model wrapper or velocity_model)
        chunk_size: Chunk size in tokens (e.g., 4096). None to disable.
    """
    # Handle X0Model wrapper
    model = transformer.velocity_model if hasattr(transformer, 'velocity_model') else transformer

    # Find and configure all BasicAVTransformerBlock instances
    count = 0
    for module in model.modules():
        if hasattr(module, 'ffn_chunk_size'):
            module.ffn_chunk_size = chunk_size
            count += 1

    if count > 0 and chunk_size is not None:
        print(f">>> FFN chunking enabled: {count} blocks with chunk_size={chunk_size}")


def phase_barrier(phase_name: str, models_to_offload: list = None, verbose: bool = True) -> None:
    """
    Ensure complete cleanup between major phases to prevent memory fragmentation.

    This function should be called at key transition points in the pipeline:
    - After text encoding (before stage 1)
    - After stage 1 denoising (before upsampling)
    - After stage 2 denoising (before VAE decoding)
    - After video decoding (before audio decoding)

    Args:
        phase_name: Description of the completed phase (for logging)
        models_to_offload: Optional list of models to move to CPU
        verbose: If True, print memory statistics
    """
    if models_to_offload:
        for model in models_to_offload:
            if model is not None:
                model.to("cpu")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    if verbose and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f">>> Phase complete: {phase_name} | GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def apply_loras_chunked_gpu(
    model: torch.nn.Module,
    lora_state_dicts: list,
    lora_strengths: list[float],
    gpu_device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """
    Apply LoRAs to a model in a chunked manner using GPU for computation.

    This function processes each layer's weight individually:
    1. Move weight and LoRA matrices to GPU
    2. Compute LoRA delta on GPU: delta = lora_B @ lora_A * strength
    3. Add delta to weight on GPU
    4. Move result back to CPU
    5. DELETE the LoRA weights from the state dict to free RAM

    This keeps peak GPU memory low while using GPU for fast computation,
    and progressively frees RAM as LoRA weights are consumed.

    Key naming conventions:
    - Model parameter: velocity_model.adaln_single.xxx.weight
    - LoRA keys (after sd_ops): adaln_single.xxx.lora_A.weight
    - So we strip "velocity_model." prefix when looking up LoRA keys
    """
    from tqdm import tqdm

    # Build a map of LoRA weights for quick lookup
    # We'll delete entries as we use them to free RAM
    lora_maps = []
    for lsd in lora_state_dicts:
        lora_map = {}
        for key in list(lsd.sd.keys()):
            if ".lora_A.weight" in key or ".lora_B.weight" in key:
                lora_map[key] = lsd.sd[key]
        lora_maps.append(lora_map)

    # Get all named parameters that might have LoRAs
    params_to_process = []
    for name, param in model.named_parameters():
        if param is not None and name.endswith(".weight"):
            # Model param: velocity_model.adaln_single.xxx.weight
            # LoRA key: adaln_single.xxx.lora_A.weight
            # Strip velocity_model. prefix if present
            lora_prefix = name[:-len(".weight")]
            if lora_prefix.startswith("velocity_model."):
                lora_prefix = lora_prefix[len("velocity_model."):]

            key_a = f"{lora_prefix}.lora_A.weight"
            key_b = f"{lora_prefix}.lora_B.weight"
            # Check if any LoRA has weights for this layer
            has_lora = any(key_a in lora_map and key_b in lora_map for lora_map in lora_maps)
            if has_lora:
                params_to_process.append((name, param, lora_prefix, key_a, key_b))

    print(f">>> Applying LoRAs to {len(params_to_process)} layers using GPU...")

    for name, param, lora_prefix, key_a, key_b in tqdm(params_to_process, desc="Applying LoRAs", miniters=100):
        # Collect all LoRA deltas for this weight
        deltas = []
        for lora_map, lsd, strength in zip(lora_maps, lora_state_dicts, lora_strengths):
            if key_a in lora_map and key_b in lora_map:
                lora_a = lora_map[key_a].to(device=gpu_device, dtype=dtype)
                lora_b = lora_map[key_b].to(device=gpu_device, dtype=dtype)
                delta = torch.matmul(lora_b * strength, lora_a)
                deltas.append(delta)
                # Free GPU memory immediately
                del lora_a, lora_b
                # FREE RAM: Delete from BOTH the map and the original state dict
                del lora_map[key_a]
                del lora_map[key_b]
                # Delete from original state dict to actually free the tensor memory
                if key_a in lsd.sd:
                    del lsd.sd[key_a]
                if key_b in lsd.sd:
                    del lsd.sd[key_b]

        if deltas:
            # Move weight to GPU, apply deltas, move back to CPU
            weight_gpu = param.data.to(device=gpu_device, dtype=dtype)
            for delta in deltas:
                weight_gpu.add_(delta)
                del delta
            param.data = weight_gpu.to(device="cpu", dtype=dtype)
            del weight_gpu

            # Periodically clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Clean up any remaining LoRA state dict references
    for lsd in lora_state_dicts:
        lsd.sd.clear()
    gc.collect()

    print(">>> LoRA application complete")


# =============================================================================
# Chunked Video Encoding
# =============================================================================

def encode_video_chunked(
    video_tensor: torch.Tensor,
    video_encoder,
    chunk_frames: int = 65,  # 8*8 + 1
    overlap_frames: int = 8,  # kept for API compatibility, but we use context_frames internally
) -> torch.Tensor:
    """
    Encode video in temporal chunks to reduce memory usage.

    Uses autoregressive chunking to handle the causal VAE encoder properly:
    - The VAE encoder uses causal convolutions where each frame depends on previous frames
    - For chunks after the first, we include context frames and discard the corresponding
      latent tokens (the "warm-up" region where the encoder lacks sufficient context)
    - This avoids blending artifacts from mismatched latents at chunk boundaries

    Args:
        video_tensor: Shape (1, C, F, H, W) normalized video
        video_encoder: VideoEncoder model
        chunk_frames: Frames per chunk (must be 8*k+1)
        overlap_frames: Legacy parameter, context_frames is used internally

    Returns:
        Encoded latent tensor (1, latent_channels, F', H', W')
    """
    _, c, total_frames, h, w = video_tensor.shape

    # If video fits in one chunk, encode directly
    if total_frames <= chunk_frames:
        return video_encoder(video_tensor)

    # Validate
    assert (chunk_frames - 1) % 8 == 0, "chunk_frames must be 8*k+1"

    # Context frames for causal encoder warm-up (must be 8*k for clean latent boundaries)
    # Using 24 frames (3 latent tokens) provides enough context for the encoder's receptive field
    context_frames = 24
    context_latents = context_frames // 8  # 3 latent tokens to discard

    # Calculate how many new frames each chunk contributes (excluding context)
    new_frames_per_chunk = chunk_frames - context_frames  # e.g., 65 - 24 = 41 frames
    new_latents_per_chunk = new_frames_per_chunk // 8  # e.g., 41 // 8 = 5 latent tokens

    # Build chunk list: (start_frame, end_frame, is_first)
    chunks_info = []

    # First chunk: starts at 0, uses all latent tokens
    first_end = min(chunk_frames, total_frames)
    chunks_info.append((0, first_end, True))

    # Subsequent chunks: start with context overlap, discard context latents
    current_frame = first_end - context_frames  # Where the next chunk's "new" content starts
    while current_frame + context_frames < total_frames:
        chunk_start = current_frame  # Include context frames
        chunk_end = min(chunk_start + chunk_frames, total_frames)
        chunks_info.append((chunk_start, chunk_end, False))
        current_frame = chunk_end - context_frames

    print(f">>> Encoding {len(chunks_info)} chunk(s) with {context_frames}-frame context...")

    # Encode chunks and collect latent segments
    latent_segments = []

    for i, (start, end, is_first) in enumerate(chunks_info):
        actual_frames = end - start

        # Pad if needed to satisfy 8*k+1
        if (actual_frames - 1) % 8 != 0:
            target = 8 * ((actual_frames - 1) // 8 + 1) + 1
            pad_frames = target - actual_frames
        else:
            pad_frames = 0

        print(f">>> Encoding chunk {i+1}/{len(chunks_info)} (frames {start}-{end})...")
        chunk = video_tensor[:, :, start:end, :, :]

        # Pad if necessary
        if pad_frames > 0:
            last_frame = chunk[:, :, -1:, :, :]
            padding = last_frame.expand(-1, -1, pad_frames, -1, -1)
            chunk = torch.cat([chunk, padding], dim=2)

        # Encode
        with torch.no_grad():
            latent = video_encoder(chunk)

        # Remove padded latent tokens if we padded
        if pad_frames > 0:
            valid_tokens = (actual_frames - 1) // 8 + 1
            latent = latent[:, :, :valid_tokens, :, :]

        # For first chunk: use all latent tokens
        # For subsequent chunks: discard context latents (warm-up region)
        if is_first:
            latent_segments.append(latent)
        else:
            # Discard the first context_latents tokens (the warm-up region)
            if latent.shape[2] > context_latents:
                latent_segments.append(latent[:, :, context_latents:, :, :])
            # If chunk is very short, it might be entirely context - skip it

        # Free memory
        del chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Concatenate all segments directly (no blending needed with this approach)
    if len(latent_segments) == 1:
        result = latent_segments[0]
    else:
        result = torch.cat(latent_segments, dim=2)

    # Trim to exact expected length
    total_latent_frames = (total_frames - 1) // 8 + 1
    if result.shape[2] > total_latent_frames:
        result = result[:, :, :total_latent_frames, :, :]
    elif result.shape[2] < total_latent_frames:
        # Pad with last token if we're short (shouldn't happen with correct chunking)
        pad_latents = total_latent_frames - result.shape[2]
        last_latent = result[:, :, -1:, :, :]
        padding = last_latent.expand(-1, -1, pad_latents, -1, -1)
        result = torch.cat([result, padding], dim=2)

    return result


# =============================================================================
# Pipeline with Offloading
# =============================================================================

class LTXVideoGeneratorWithOffloading:
    """
    LTX-2 video generator with intelligent model offloading.

    This class wraps the two-stage pipeline and manages model lifecycle
    to minimize GPU memory usage through strategic CPU offloading.
    """

    def __init__(
        self,
        checkpoint_path: str,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        stage2_loras: list[LoraPathStrengthAndSDOps] = None,
        device: str = None,
        fp8transformer: bool = False,
        offload: bool = False,
        # Separate block swap controls
        enable_dit_block_swap: bool = False,
        dit_blocks_in_memory: int = 22,
        enable_text_encoder_block_swap: bool = False,
        text_encoder_blocks_in_memory: int = 6,
        enable_refiner_block_swap: bool = False,
        refiner_blocks_in_memory: int = 22,
        enable_activation_offload: bool = False,
        temporal_chunk_size: int = 0,
        one_stage: bool = False,
        refine_only: bool = False,
        distilled_checkpoint: bool = False,
        stage2_checkpoint: str | None = None,
        ffn_chunk_size: int | None = None,
        vae_path: str | None = None,
    ):
        self.device = device or get_device()
        self.dtype = torch.bfloat16
        self.offload = offload
        # Separate block swap settings
        self.enable_dit_block_swap = enable_dit_block_swap
        self.dit_blocks_in_memory = dit_blocks_in_memory
        self.enable_text_encoder_block_swap = enable_text_encoder_block_swap
        self.text_encoder_blocks_in_memory = text_encoder_blocks_in_memory
        self.enable_refiner_block_swap = enable_refiner_block_swap
        self.refiner_blocks_in_memory = refiner_blocks_in_memory
        self.enable_activation_offload = enable_activation_offload
        self.temporal_chunk_size = temporal_chunk_size
        self.one_stage = one_stage
        self.refine_only = refine_only
        self.distilled_checkpoint = distilled_checkpoint
        self.stage2_checkpoint = stage2_checkpoint
        self.ffn_chunk_size = ffn_chunk_size
        self.vae_path = vae_path

        # Create model ledger for stage 1
        self.stage_1_model_ledger = ModelLedger(
            dtype=self.dtype,
            device=self.device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            spatial_upsampler_path=spatial_upsampler_path,
            vae_path=vae_path,
            loras=loras,
            fp8transformer=fp8transformer,
        )

        # Store params for stage 2 (create fresh ledger later to avoid shared state issues)
        # Use stage2_checkpoint if provided, otherwise use the main checkpoint
        self._stage_2_checkpoint_path = stage2_checkpoint if stage2_checkpoint else checkpoint_path
        self._stage_2_gemma_root = gemma_root
        self._stage_2_spatial_upsampler_path = spatial_upsampler_path
        self._stage_2_vae_path = vae_path
        # Stage 2 gets stage2_loras (user LoRAs for stage 2 only), not the stage 1 loras
        self._stage_2_loras = stage2_loras or []
        self._stage_2_distilled_lora = distilled_lora
        self._stage_2_fp8transformer = fp8transformer

        # Create model ledger for stage 2 (with distilled LoRA)
        # Note: Creating via with_loras for now, will create fresh in generate if needed
        self.stage_2_model_ledger = self.stage_1_model_ledger.with_loras(
            loras=distilled_lora,
        )

        # Pipeline components (patchifiers, scale factors)
        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=self.device,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        cfg_guidance_scale: float,
        images: list[tuple[str, int, float]],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        disable_audio: bool = False,
        input_video: str | None = None,
        refine_strength: float = 0.3,
        refine_steps: int = 10,
        stage2_steps: int = 3,
        anchor_image: str | None = None,
        anchor_interval: int | None = None,
        anchor_strength: float = 0.8,
        anchor_decay: str | None = None,
        audio: str | None = None,
        audio_strength: float = 1.0,
        # STG (Spatio-Temporal Guidance) parameters
        stg_scale: float = 1.0,
        stg_blocks: list[int] | None = None,
        stg_mode: str = "stg_av",
        # SVI Pro parameters
        _motion_latent: torch.Tensor | None = None,
        _num_motion_latent: int = 0,
        # Sliding window overlap parameters
        _overlap_latent: torch.Tensor | None = None,
        _num_overlap_latent: int = 0,
        _overlap_strength: float = 0.95,
        # Preview callback
        preview_callback: Callable | None = None,
        preview_callback_interval: int = 1,
        # Depth Control (IC-LoRA) parameters
        depth_video: str | None = None,
        depth_image: str | None = None,
        estimate_depth: bool = False,
        depth_strength: float = 1.0,
        depth_stage2: bool = False,
        # Latent normalization (fixes overbaking/audio clipping)
        latent_norm_fn: Callable | None = None,
        # V2A mode (video-to-audio: freeze video, generate audio)
        v2a_mode: bool = False,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor | None, str | None]:
        """
        Generate video with optional audio.

        Returns:
            Tuple of (video_iterator, audio_tensor or None, enhanced_prompt or None)
        """
        # Validate resolution
        assert_resolution(height=height, width=width, is_two_stage=not self.one_stage)

        # Initialize diffusion components
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        cfg_guider = CFGGuider(cfg_guidance_scale)
        # Initialize STG (Spatio-Temporal Guidance) components
        effective_stg_blocks = stg_blocks if stg_blocks is not None else [29]
        stg_guider = STGGuider(stg_scale)
        stg_perturbation_config = build_stg_perturbation_config(
            stg_scale=stg_scale,
            stg_blocks=effective_stg_blocks,
            stg_mode=stg_mode,
        )
        dtype = self.dtype

        start_time = time.time()

        # =====================================================================
        # Phase 1: Text Encoding
        # =====================================================================
        print(">>> Loading text encoder...", flush=True)
        text_encoder_block_swap = None
        if self.enable_text_encoder_block_swap:
            # Load text encoder to CPU first for block swapping
            original_device = self.stage_1_model_ledger.device
            self.stage_1_model_ledger.device = torch.device("cpu")
            text_encoder = self.stage_1_model_ledger.text_encoder()
            self.stage_1_model_ledger.device = original_device

            # Enable block swap for text encoder
            print(f">>> Enabling text encoder block swap ({self.text_encoder_blocks_in_memory} layers in GPU)...", flush=True)
            text_encoder_block_swap = enable_text_encoder_block_swap(
                text_encoder,
                blocks_in_memory=self.text_encoder_blocks_in_memory,
                device=self.device,
            )
        else:
            text_encoder = self.stage_1_model_ledger.text_encoder()

        # Track the enhanced prompt for metadata (None if not enhanced)
        enhanced_prompt = None
        if enhance_prompt:
            print(">>> Enhancing prompt with Gemma...")
            prompt = generate_enhanced_prompt(
                text_encoder, prompt, images[0][0] if len(images) > 0 else None, seed=seed
            )
            enhanced_prompt = prompt
            print(f">>> Enhanced prompt: {prompt}")

        print(">>> Encoding prompts...")
        # Encode both prompts if CFG might be used (cfg_guidance_scale > 1)
        # Otherwise only encode positive prompt to save memory/time
        if cfg_guidance_scale > 1.0:
            # Encode both positive and negative prompts for CFG
            context_p, context_n = encode_text(text_encoder, prompts=[prompt, negative_prompt])
            v_context_p, a_context_p = context_p
            v_context_n, a_context_n = context_n
        else:
            # Only encode positive prompt (no CFG needed)
            context_p = encode_text(text_encoder, prompts=[prompt])[0]
            v_context_p, a_context_p = context_p
            v_context_n, a_context_n = None, None

        # Offload text encoder
        print(">>> Releasing text encoder from GPU...")
        if text_encoder_block_swap:
            offload_all_text_encoder_blocks(text_encoder)
            text_encoder_block_swap = None
        else:
            text_encoder.to("cpu")
        del text_encoder
        synchronize_and_cleanup()

        # Move text embeddings to CPU - they'll be moved back to GPU when needed for denoising
        # This frees GPU memory during model loading phases
        v_context_p = v_context_p.cpu()
        a_context_p = a_context_p.cpu()
        if v_context_n is not None:
            v_context_n = v_context_n.cpu()
        if a_context_n is not None:
            a_context_n = a_context_n.cpu()

        print(f">>> Text encoding completed in {time.time() - start_time:.1f}s")
        phase_barrier("text_encoding")

        # Initialize audio_latent for use in stage 2
        # Will be set by refine-only mode if encoding audio from input video
        audio_latent = None

        # =====================================================================
        # Refine-only mode: Skip stage 1 and use input video directly
        # =====================================================================
        # Store keyframe conditionings for refine-only mode (populated below if needed)
        refine_keyframe_conditionings = []

        # Initialize audio_conditionings for all code paths
        audio_conditionings = []

        if self.refine_only and input_video:
            print(">>> Refine-only mode: Loading input video and creating keyframe conditionings...")
            refine_start = time.time()

            video_encoder = self.stage_1_model_ledger.video_encoder()

            # Load input video
            video_tensor = load_video_conditioning(
                video_path=input_video,
                height=height,
                width=width,
                frame_cap=num_frames,
                dtype=dtype,
                device=self.device,
            )

            # Use keyframe conditioning approach (like Wan2GP) instead of encoding whole video
            # This avoids chunked encoding glitches from the causal VAE encoder
            # Extract every 8th frame and create conditionings with strength=1.0
            latent_stride = 8
            actual_frames = video_tensor.shape[2]

            # Get latent indices already covered by --image args (don't duplicate conditioning)
            image_latent_indices = set()
            if images:
                for _, frame_idx, _ in images:
                    image_latent_indices.add(frame_idx // latent_stride)

            print(f">>> Creating keyframe conditionings for {actual_frames} frames (every {latent_stride} frames)...")
            if image_latent_indices:
                print(f">>> Skipping latent indices {sorted(image_latent_indices)} (covered by --image)")

            from ltx_core.conditioning import VideoConditionByLatentIndex
            for frame_idx in range(0, actual_frames, latent_stride):
                latent_idx = frame_idx // latent_stride

                # Skip frames already covered by --image conditionings
                if latent_idx in image_latent_indices:
                    continue

                # Extract single frame: video_tensor is [1, C, F, H, W]
                frame = video_tensor[:, :, frame_idx:frame_idx+1, :, :]  # [1, C, 1, H, W]

                # Encode frame to latent
                with torch.no_grad():
                    encoded_frame = video_encoder(frame)

                # Create conditioning with strength=1.0 to preserve frame exactly
                refine_keyframe_conditionings.append(
                    VideoConditionByLatentIndex(
                        latent=encoded_frame,
                        strength=1.0,
                        latent_idx=latent_idx,
                    )
                )

            print(f">>> Created {len(refine_keyframe_conditionings)} keyframe conditionings")

            # No initial video latent - using keyframe conditionings instead
            upscaled_video_latent = None

            # Extract and encode audio from input video (like stage 1 would)
            audio_latent = None
            if not disable_audio:
                print(">>> Encoding audio from input video...")

                # Extract audio waveform from input video
                waveform, sample_rate = decode_audio_from_file(input_video, self.device)

                if waveform is not None:
                    audio_encoder = self.stage_1_model_ledger.audio_encoder()

                    # Create audio processor with encoder's parameters
                    audio_processor = AudioProcessor(
                        sample_rate=audio_encoder.sample_rate,
                        mel_bins=audio_encoder.mel_bins,
                        mel_hop_length=audio_encoder.mel_hop_length,
                        n_fft=audio_encoder.n_fft,
                    ).to(self.device)

                    # Reshape waveform to [batch, channels, total_samples]
                    # decode_audio_from_file returns [1, channels, total_samples]
                    if waveform.dim() == 3:
                        # Flatten frames into samples: [num_frames, channels, samples] -> [1, channels, total_samples]
                        num_frames_audio, channels, samples_per_frame = waveform.shape
                        waveform = waveform.permute(1, 0, 2).reshape(channels, -1).unsqueeze(0)
                    elif waveform.dim() == 2:
                        # [channels, samples] -> [1, channels, samples]
                        waveform = waveform.unsqueeze(0)

                    # Convert waveform to mel spectrogram (use float32 for audio quality)
                    mel_spectrogram = audio_processor.waveform_to_mel(
                        waveform.to(dtype=torch.float32),
                        waveform_sample_rate=sample_rate
                    )

                    # Encode mel spectrogram to latents
                    audio_latent = audio_encoder(mel_spectrogram.to(dtype=torch.float32))
                    # Convert to bfloat16 for consistency with pipeline
                    audio_latent = audio_latent.to(dtype=dtype)

                    # Clean up audio encoder
                    del audio_encoder, audio_processor
                    cleanup_memory()
                    print(">>> Audio encoded successfully")
                else:
                    print(">>> Input video has no audio track, will generate fresh audio")

            print(f">>> Input video encoded in {time.time() - refine_start:.1f}s")

            # Clean up video encoder before loading stage 2
            del video_encoder
            cleanup_memory()

            # Skip Phase 2, 3 - go directly to Phase 4 (Stage 2)
            block_swap_manager = None
            video_encoder = None  # Not needed for refine-only

        # =====================================================================
        # Phase 2: Stage 1 - Low Resolution Generation (skip for refine-only)
        # =====================================================================
        skip_stage_1 = self.refine_only and input_video
        if not skip_stage_1:
            print(">>> Stage 1: Loading video encoder and transformer...")
            stage1_start = time.time()

            video_encoder = self.stage_1_model_ledger.video_encoder()

            # For block swapping, load transformer to CPU first, then selectively move blocks
            block_swap_manager = None
            if self.enable_dit_block_swap:
                print(f">>> Loading DiT transformer for block swapping...", flush=True)
                # Check if there are LoRAs to apply
                has_loras = hasattr(self.stage_1_model_ledger, 'loras') and self.stage_1_model_ledger.loras

                if has_loras:
                    # Use chunked GPU LoRA application (like stage 2) to avoid slow CPU computation
                    # 1. Create a ledger WITHOUT LoRAs to load base model quickly
                    stage_1_ledger_no_lora = ModelLedger(
                        dtype=self.dtype,
                        device=torch.device("cpu"),
                        checkpoint_path=self.stage_1_model_ledger.checkpoint_path,
                        gemma_root_path=self.stage_1_model_ledger.gemma_root_path,
                        spatial_upsampler_path=self.stage_1_model_ledger.spatial_upsampler_path,
                        vae_path=self.stage_1_model_ledger.vae_path,
                        loras=(),  # No LoRAs - load base model only
                        fp8transformer=self.stage_1_model_ledger.fp8transformer,
                    )

                    # 2. Load transformer without LoRAs (fast)
                    print(">>> Loading base transformer to CPU...", flush=True)
                    transformer = stage_1_ledger_no_lora.transformer()

                    # 3. Apply LoRAs using chunked GPU computation
                    loras = self.stage_1_model_ledger.loras
                    print(f">>> Applying {len(loras)} LoRA(s) using chunked GPU computation...", flush=True)
                    from ltx_core.loader.sft_loader import SafetensorsStateDictLoader
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
                        gpu_device=self.device,
                        dtype=self.dtype,
                    )

                    # Clean up LoRA state dicts
                    del lora_state_dicts
                    synchronize_and_cleanup()
                else:
                    # No LoRAs - load directly to CPU
                    original_device = self.stage_1_model_ledger.device
                    self.stage_1_model_ledger.device = torch.device("cpu")
                    transformer = self.stage_1_model_ledger.transformer()
                    self.stage_1_model_ledger.device = original_device

                # Move non-block components to GPU, keep blocks on CPU
                print(f">>> Enabling DiT block swapping ({self.dit_blocks_in_memory} blocks in GPU)...")
                # Move the wrapper and non-block parts to GPU
                transformer.velocity_model.patchify_proj.to(self.device)
                transformer.velocity_model.adaln_single.to(self.device)
                transformer.velocity_model.caption_projection.to(self.device)
                transformer.velocity_model.norm_out.to(self.device)
                transformer.velocity_model.proj_out.to(self.device)
                transformer.velocity_model.scale_shift_table = torch.nn.Parameter(
                    transformer.velocity_model.scale_shift_table.to(self.device)
                )
                # Audio components
                if hasattr(transformer.velocity_model, "audio_patchify_proj"):
                    transformer.velocity_model.audio_patchify_proj.to(self.device)
                if hasattr(transformer.velocity_model, "audio_adaln_single"):
                    transformer.velocity_model.audio_adaln_single.to(self.device)
                if hasattr(transformer.velocity_model, "audio_caption_projection"):
                    transformer.velocity_model.audio_caption_projection.to(self.device)
                if hasattr(transformer.velocity_model, "audio_norm_out"):
                    transformer.velocity_model.audio_norm_out.to(self.device)
                if hasattr(transformer.velocity_model, "audio_proj_out"):
                    transformer.velocity_model.audio_proj_out.to(self.device)
                if hasattr(transformer.velocity_model, "audio_scale_shift_table"):
                    transformer.velocity_model.audio_scale_shift_table = torch.nn.Parameter(
                        transformer.velocity_model.audio_scale_shift_table.to(self.device)
                    )
                # Cross-attention adaln components
                if hasattr(transformer.velocity_model, "av_ca_video_scale_shift_adaln_single"):
                    transformer.velocity_model.av_ca_video_scale_shift_adaln_single.to(self.device)
                if hasattr(transformer.velocity_model, "av_ca_audio_scale_shift_adaln_single"):
                    transformer.velocity_model.av_ca_audio_scale_shift_adaln_single.to(self.device)
                if hasattr(transformer.velocity_model, "av_ca_a2v_gate_adaln_single"):
                    transformer.velocity_model.av_ca_a2v_gate_adaln_single.to(self.device)
                if hasattr(transformer.velocity_model, "av_ca_v2a_gate_adaln_single"):
                    transformer.velocity_model.av_ca_v2a_gate_adaln_single.to(self.device)

                # Use activation offload for extreme memory savings (moves activations to CPU between blocks)
                if self.enable_activation_offload:
                    block_swap_manager = enable_block_swap_with_activation_offload(
                        transformer,
                        blocks_in_memory=self.dit_blocks_in_memory,
                        device=self.device,
                        verbose=True,
                        temporal_chunk_size=self.temporal_chunk_size,
                    )
                else:
                    block_swap_manager = enable_block_swap(
                        transformer,
                        blocks_in_memory=self.dit_blocks_in_memory,
                        device=self.device,
                    )
            else:
                transformer = self.stage_1_model_ledger.transformer()

            # Enable FFN chunking for long sequences if configured
            if self.ffn_chunk_size is not None:
                set_ffn_chunk_size(transformer, self.ffn_chunk_size)

            # Create diffusion schedule
            # Both distilled and standard checkpoints use configurable LTX2Scheduler
            sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(
                dtype=torch.float32, device=self.device
            )
            print(f">>> Using {num_inference_steps} inference steps")

            # Move text embeddings to GPU for stage 1 denoising
            v_context_p = v_context_p.to(self.device)
            a_context_p = a_context_p.to(self.device)
            if v_context_n is not None:
                v_context_n = v_context_n.to(self.device)
            if a_context_n is not None:
                a_context_n = a_context_n.to(self.device)

            # Define denoising function for stage 1
            # Use CFG+STG guidance if either is enabled, otherwise simple denoising
            use_cfg = cfg_guidance_scale > 1.0 and v_context_n is not None
            use_stg = stg_guider.enabled() and stg_perturbation_config is not None
            # Convert anchor_decay "none" to None for the denoising loop
            effective_anchor_decay = anchor_decay if anchor_decay and anchor_decay != "none" else None
            if use_cfg or use_stg:
                # CFG+STG guidance (handles both independently)
                def first_stage_denoising_loop(
                    sigmas: torch.Tensor,
                    video_state: LatentState,
                    audio_state: LatentState,
                    stepper: DiffusionStepProtocol,
                ) -> tuple[LatentState, LatentState]:
                    return euler_denoising_loop(
                        sigmas=sigmas,
                        video_state=video_state,
                        audio_state=audio_state,
                        stepper=stepper,
                        denoise_fn=cfg_stg_denoising_func(
                            cfg_guider=cfg_guider,
                            stg_guider=stg_guider,
                            stg_perturbation_config=stg_perturbation_config,
                            v_context_p=v_context_p,
                            v_context_n=v_context_n,
                            a_context_p=a_context_p,
                            a_context_n=a_context_n,
                            transformer=transformer,
                        ),
                        anchor_decay=effective_anchor_decay,
                        callback=preview_callback,
                        callback_interval=preview_callback_interval,
                        latent_norm_fn=latent_norm_fn,
                    )
            else:
                # No guidance, single forward pass
                def first_stage_denoising_loop(
                    sigmas: torch.Tensor,
                    video_state: LatentState,
                    audio_state: LatentState,
                    stepper: DiffusionStepProtocol,
                ) -> tuple[LatentState, LatentState]:
                    return euler_denoising_loop(
                        sigmas=sigmas,
                        video_state=video_state,
                        audio_state=audio_state,
                        stepper=stepper,
                        denoise_fn=simple_denoising_func(
                            video_context=v_context_p,
                            audio_context=a_context_p,
                            transformer=transformer,
                        ),
                        anchor_decay=effective_anchor_decay,
                        callback=preview_callback,
                        callback_interval=preview_callback_interval,
                        latent_norm_fn=latent_norm_fn,
                    )

            # Stage 1 output shape (half resolution for two-stage, full for one-stage)
            if self.one_stage:
                stage_1_output_shape = VideoPixelShape(
                    batch=1,
                    frames=num_frames,
                    width=width,
                    height=height,
                    fps=frame_rate,
                )
            else:
                stage_1_output_shape = VideoPixelShape(
                    batch=1,
                    frames=num_frames,
                    width=width // 2,
                    height=height // 2,
                    fps=frame_rate,
                )

            # Image conditioning for stage 1 (i2v - replaces latent at frame 0)
            stage_1_conditionings = image_conditionings_by_replacing_latent(
                images=images,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
            )

            # Anchor image conditioning for stage 1 (guiding latent - appends tokens)
            if anchor_interval is not None:
                anchor_tuples = build_anchor_image_tuples(
                    anchor_image=anchor_image,
                    anchor_interval=anchor_interval,
                    anchor_strength=anchor_strength,
                    num_frames=num_frames,
                    images=images,
                )
                if anchor_tuples:
                    anchor_conditionings = image_conditionings_by_adding_guiding_latent(
                        images=anchor_tuples,
                        height=stage_1_output_shape.height,
                        width=stage_1_output_shape.width,
                        video_encoder=video_encoder,
                        dtype=dtype,
                        device=self.device,
                    )
                    stage_1_conditionings = stage_1_conditionings + anchor_conditionings
                    print(f">>> Added {len(anchor_conditionings)} anchor points at frames {[t[1] for t in anchor_tuples]}")

            # V2V: Add video conditioning from input video (like ic_lora pipeline)
            # Uses tiled encoding to handle long videos without OOM
            v2v_audio_latent = None  # Will be set if input_video has audio
            if input_video and not self.refine_only:
                from ltx_core.conditioning import VideoConditionByKeyframeIndex
                print(f">>> V2V: Loading and encoding input video with tiled encoding...")
                v2v_start = time.time()

                # Load input video at stage 1 resolution - keep on CPU to save GPU memory
                video_tensor = load_video_conditioning(
                    video_path=input_video,
                    height=stage_1_output_shape.height,
                    width=stage_1_output_shape.width,
                    frame_cap=num_frames,
                    dtype=dtype,
                    device=torch.device("cpu"),
                )

                # Tiled encoding parameters (matching decoder's default temporal tiling)
                # tile_size must be 8k+1 for valid VAE input
                # Use small tiles since transformer is already in GPU memory
                tile_size = 17  # 2 latent frames + 1 - minimal size for memory safety
                tile_overlap = 8  # Overlap in pixel frames (divisible by 8)
                tile_stride = tile_size - tile_overlap  # Non-overlapping portion

                actual_frames = video_tensor.shape[2]
                num_chunks = 0

                # Encode video in overlapping tiles
                frame_idx = 0
                while frame_idx < actual_frames:
                    # Calculate chunk bounds
                    chunk_end = min(frame_idx + tile_size, actual_frames)
                    chunk_frames = chunk_end - frame_idx

                    # Ensure valid frame count (8k+1) for VAE
                    valid_frames = ((chunk_frames - 1) // 8) * 8 + 1
                    if valid_frames < 9:  # Need at least 9 frames for meaningful encoding
                        break
                    chunk_end = frame_idx + valid_frames

                    # Extract chunk and move to GPU for encoding
                    chunk = video_tensor[:, :, frame_idx:chunk_end, :, :].to(device=self.device, dtype=dtype)
                    with torch.no_grad():
                        encoded_chunk = video_encoder(chunk)
                    del chunk
                    torch.cuda.empty_cache()

                    # Move encoded latent to CPU to free GPU memory
                    encoded_chunk_cpu = encoded_chunk.cpu()
                    del encoded_chunk
                    torch.cuda.empty_cache()

                    # Add as keyframe conditioning with frame offset
                    stage_1_conditionings.append(
                        VideoConditionByKeyframeIndex(
                            keyframes=encoded_chunk_cpu,
                            frame_idx=frame_idx,
                            strength=refine_strength,
                        )
                    )
                    num_chunks += 1

                    # Move to next tile (with overlap)
                    frame_idx += tile_stride
                    if frame_idx >= actual_frames - 8:  # Don't create tiny final chunks
                        break

                # Clean up video tensor
                del video_tensor

                print(f">>> V2V: Added {num_chunks} video conditioning chunks (strength={refine_strength}) in {time.time() - v2v_start:.1f}s")

                # Extract and encode audio from input video to preserve it
                if not disable_audio:
                    print(">>> V2V: Extracting audio from input video...")
                    waveform, sample_rate = decode_audio_from_file(input_video, self.device)

                    if waveform is not None:
                        # Calculate expected audio duration and trim/pad waveform to match
                        expected_duration = float(num_frames) / float(frame_rate)
                        expected_samples = int(expected_duration * sample_rate)

                        # Trim waveform if longer than expected
                        if waveform.shape[-1] > expected_samples:
                            waveform = waveform[..., :expected_samples]
                            print(f">>> V2V: Trimmed audio to {expected_samples} samples ({expected_duration:.3f}s)")
                        elif waveform.shape[-1] < expected_samples:
                            # Pad waveform if shorter than expected
                            padding = expected_samples - waveform.shape[-1]
                            waveform = torch.nn.functional.pad(waveform, (0, padding))
                            print(f">>> V2V: Padded audio by {padding} samples to {expected_samples} total ({expected_duration:.3f}s)")

                        audio_encoder = self.stage_1_model_ledger.audio_encoder()

                        audio_processor = AudioProcessor(
                            sample_rate=audio_encoder.sample_rate,
                            mel_bins=audio_encoder.mel_bins,
                            mel_hop_length=audio_encoder.mel_hop_length,
                            n_fft=audio_encoder.n_fft,
                        ).to(self.device)

                        if waveform.dim() == 3:
                            num_frames_audio, channels, samples_per_frame = waveform.shape
                            waveform = waveform.permute(1, 0, 2).reshape(channels, -1).unsqueeze(0)
                        elif waveform.dim() == 2:
                            waveform = waveform.unsqueeze(0)

                        mel_spectrogram = audio_processor.waveform_to_mel(
                            waveform.to(dtype=torch.float32),
                            waveform_sample_rate=sample_rate
                        )

                        v2v_audio_latent = audio_encoder(mel_spectrogram.to(dtype=torch.float32))
                        v2v_audio_latent = v2v_audio_latent.to(dtype=dtype)

                        del audio_encoder, audio_processor
                        cleanup_memory()
                        print(">>> V2V: Audio extracted and encoded successfully")
                    else:
                        v2v_audio_latent = None
                        print(">>> V2V: Input video has no audio track")
                else:
                    v2v_audio_latent = None

            # V2A Mode: Freeze entire video, generate fresh audio
            if v2a_mode and input_video:
                from ltx_core.conditioning import VideoConditionByLatentIndex
                print(f">>> V2A Mode: Encoding input video (video will be frozen)...")
                v2a_start = time.time()

                # Load input video at stage 1 resolution
                video_tensor = load_video_conditioning(
                    video_path=input_video,
                    height=stage_1_output_shape.height,
                    width=stage_1_output_shape.width,
                    frame_cap=num_frames,
                    dtype=dtype,
                    device=torch.device("cpu"),
                )

                actual_frames = video_tensor.shape[2]
                latent_stride = 8  # VAE temporal compression factor

                # Encode each latent frame position (every 8th pixel frame)
                for frame_idx in range(0, actual_frames, latent_stride):
                    latent_idx = frame_idx // latent_stride
                    frame = video_tensor[:, :, frame_idx:frame_idx+1, :, :].to(device=self.device, dtype=dtype)

                    with torch.no_grad():
                        encoded_frame = video_encoder(frame)

                    # strength=1.0 → denoise_mask=0 (frozen)
                    stage_1_conditionings.append(
                        VideoConditionByLatentIndex(
                            latent=encoded_frame.cpu(),
                            strength=1.0,
                            latent_idx=latent_idx,
                        )
                    )
                    del frame, encoded_frame
                    torch.cuda.empty_cache()

                del video_tensor
                cleanup_memory()

                num_latent_frames = (actual_frames - 1) // latent_stride + 1
                print(f">>> V2A: {num_latent_frames} video frames frozen (stage 1) in {time.time() - v2a_start:.1f}s")

                # Clear any V2V audio - we want fresh audio generation
                v2v_audio_latent = None
                print(">>> V2A: Audio will be generated fresh (not extracted from input)")

            # Depth Control: Add depth conditioning for IC-LoRA
            depth_tensor = None
            if depth_video or depth_image or estimate_depth:
                print(f">>> Depth Control: {'Estimating' if estimate_depth else 'Loading'} depth maps...")
                depth_tensor = load_or_estimate_depth(
                    depth_video=depth_video,
                    depth_image=depth_image,
                    estimate_depth=estimate_depth,
                    source_video=input_video,
                    source_image=images[0][0] if images else None,
                    num_frames=num_frames,
                    height=stage_1_output_shape.height,
                    width=stage_1_output_shape.width,
                    device=self.device,
                    dtype=dtype,
                )
                if depth_tensor is not None:
                    depth_conditionings = encode_depth_conditioning(
                        depth_tensor=depth_tensor,
                        video_encoder=video_encoder,
                        num_frames=num_frames,
                        strength=depth_strength,
                        device=self.device,
                        dtype=dtype,
                    )
                    stage_1_conditionings = stage_1_conditionings + depth_conditionings
                    print(f">>> Depth Control: Added {len(depth_conditionings)} depth conditioning chunks (strength={depth_strength})")

            # SVI Pro: Add motion latent conditionings
            if _motion_latent is not None and _num_motion_latent > 0:
                motion_cond = VideoConditionByMotionLatent(
                    motion_latent=_motion_latent,
                    strength=0.7,
                    frame_offset=0,
                )
                motion_conditionings = motion_cond.to_keyframe_conditionings(
                    num_frames=num_frames,
                    dtype=dtype,
                    device=self.device,
                )
                stage_1_conditionings = stage_1_conditionings + motion_conditionings
                print(f">>> SVI Pro: Added {len(motion_conditionings)} motion keyframe conditionings")

            # Sliding Window: Add overlap latent conditionings at the BEGINNING of the sequence
            if _overlap_latent is not None and _num_overlap_latent > 0:
                from ltx_core.conditioning import VideoConditionByLatentIndex
                overlap_latent_tensor = _overlap_latent.to(device=self.device, dtype=dtype)
                num_overlap_frames = overlap_latent_tensor.shape[2]

                for i in range(min(num_overlap_frames, _num_overlap_latent)):
                    frame_latent = overlap_latent_tensor[:, :, i:i+1, :, :]
                    stage_1_conditionings.append(
                        VideoConditionByLatentIndex(
                            latent=frame_latent,
                            strength=_overlap_strength,
                            latent_idx=i,  # Frame 0, 1, 2... at the beginning
                        )
                    )
                print(f">>> Sliding Window: Injected {min(num_overlap_frames, _num_overlap_latent)} overlap latent frames at beginning")

            audio_conditionings = []
            if audio is not None:
                from ltx_core.conditioning import AudioConditionByLatent
                print(f">>> Loading and encoding audio from {audio}...")
                waveform, sample_rate = decode_audio_from_file(audio, self.device)
                if waveform is not None:
                    audio_encoder = self.stage_1_model_ledger.audio_encoder()
                    audio_processor = AudioProcessor(
                        sample_rate=audio_encoder.sample_rate,
                        mel_bins=audio_encoder.mel_bins,
                        mel_hop_length=audio_encoder.mel_hop_length,
                        n_fft=audio_encoder.n_fft,
                    ).to(self.device)

                    if waveform.dim() == 3:
                        num_frames_audio, channels, samples_per_frame = waveform.shape
                        waveform = waveform.permute(1, 0, 2).reshape(channels, -1).unsqueeze(0)
                    elif waveform.dim() == 2:
                        waveform = waveform.unsqueeze(0)

                    mel_spectrogram = audio_processor.waveform_to_mel(
                        waveform.to(dtype=torch.float32),
                        waveform_sample_rate=sample_rate
                    )
                    audio_latent = audio_encoder(mel_spectrogram.to(dtype=torch.float32))
                    audio_latent = audio_latent.to(dtype=dtype)

                    audio_conditionings.append(
                        AudioConditionByLatent(
                            latent=audio_latent,
                            strength=audio_strength,
                        )
                    )

                    del audio_encoder, audio_processor
                    cleanup_memory()
                    print(f">>> Audio encoded with strength {audio_strength} (sample rate: {sample_rate}Hz)")
                else:
                    print(">>> Warning: Could not load audio file")

            stage_label = "One-stage" if self.one_stage else "Stage 1"
            print(f">>> {stage_label}: Generating at {stage_1_output_shape.width}x{stage_1_output_shape.height}...")
            video_state, audio_state = denoise_audio_video(
                output_shape=stage_1_output_shape,
                conditionings=stage_1_conditionings,
                noiser=noiser,
                sigmas=sigmas,
                stepper=stepper,
                denoising_loop_fn=first_stage_denoising_loop,
                components=self.pipeline_components,
                dtype=dtype,
                device=self.device,
                audio_conditionings=audio_conditionings if audio_conditionings else None,
            )

            print(f">>> {stage_label} completed in {time.time() - stage1_start:.1f}s", flush=True)

            # Cleanup stage 1 transformer
            if block_swap_manager:
                offload_all_blocks(transformer)
                # Clear offloader references from transformer to break reference cycle
                if hasattr(transformer, 'velocity_model'):
                    if hasattr(transformer.velocity_model, '_block_swap_offloader'):
                        transformer.velocity_model._block_swap_offloader = None
                    if hasattr(transformer.velocity_model, '_blocks_ref'):
                        transformer.velocity_model._blocks_ref = None
                if hasattr(transformer, '_block_swap_offloader'):
                    transformer._block_swap_offloader = None
                if hasattr(transformer, '_blocks_ref'):
                    transformer._blocks_ref = None
                block_swap_manager = None
            if self.offload:
                print(">>> Offloading stage 1 transformer to CPU...")
            # Set to None instead of del to avoid GC issues
            transformer = None
            cleanup_memory()

            # Move text embeddings to CPU to free GPU memory during upsampling and stage 2 model loading
            v_context_p = v_context_p.cpu()
            a_context_p = a_context_p.cpu()
            if v_context_n is not None:
                v_context_n = v_context_n.cpu()
            if a_context_n is not None:
                a_context_n = a_context_n.cpu()

            # For one-stage, skip upsampling and stage 2 refinement
            if self.one_stage:
                # Cleanup video encoder
                video_encoder = None
                cleanup_memory()

                # Skip directly to VAE decoding (sequential loading to minimize peak memory)
                print(">>> Decoding video...")
                decode_start = time.time()

                # Load video decoder, decode, then offload
                video_decoder = self.stage_1_model_ledger.video_decoder()
                # vae_decode_video returns a generator - consume it immediately before offloading decoder
                decoded_video_chunks = []
                for chunk in vae_decode_video(
                    video_state.latent,
                    video_decoder,
                    tiling_config,
                ):
                    decoded_video_chunks.append(chunk.cpu())  # Move chunks to CPU immediately
                    del chunk
                # Create a generator that yields the cached chunks for encode_video compatibility
                def decoded_video_iter():
                    for chunk in decoded_video_chunks:
                        yield chunk.cuda()  # Move back to GPU when consumed
                decoded_video = decoded_video_iter()
                video_decoder.to("cpu")
                del video_decoder
                synchronize_and_cleanup()
                phase_barrier("video_decoding_one_stage")

                if not disable_audio:
                    print(">>> Decoding audio...")
                    # Load audio models only after video decoder is offloaded
                    audio_decoder = self.stage_1_model_ledger.audio_decoder()
                    vocoder = self.stage_1_model_ledger.vocoder()
                    decoded_audio = vae_decode_audio(
                        audio_state.latent,
                        audio_decoder,
                        vocoder,
                    )
                    audio_decoder.to("cpu")
                    vocoder.to("cpu")
                    del audio_decoder, vocoder
                    synchronize_and_cleanup()
                else:
                    decoded_audio = None

                print(f">>> Decoding completed in {time.time() - decode_start:.1f}s")
                print(f">>> Total generation time: {time.time() - start_time:.1f}s")

                return decoded_video, decoded_audio, enhanced_prompt

            # =====================================================================
            # Phase 3: Spatial Upsampling (two-stage only)
            # =====================================================================
            print(">>> Upsampling latents (2x)...", flush=True)
            upsample_start = time.time()

            spatial_upsampler = self.stage_2_model_ledger.spatial_upsampler()
            upscaled_video_latent = upsample_video(
                latent=video_state.latent[:1],
                video_encoder=video_encoder,
                upsampler=spatial_upsampler,
            )

            torch.cuda.synchronize()
            cleanup_memory()
            print(f">>> Upsampling completed in {time.time() - upsample_start:.1f}s", flush=True)
        # End of skip_stage_1 block

        # =====================================================================
        # Phase 4: Stage 2 - High Resolution Refinement (two-stage only)
        # =====================================================================
        print(">>> Stage 2: Loading transformer with distilled LoRA...", flush=True)
        stage2_start = time.time()

        # Force complete cleanup before loading stage 2 transformer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # For block swapping with LoRAs, we need to:
        # 1. Load transformer WITHOUT LoRAs to CPU (fast)
        # 2. Load LoRA state dicts
        # 3. Apply LoRAs using chunked GPU computation (fast, low memory)
        # This avoids the slow CPU-only LoRA application that appears to hang.
        block_swap_manager = None
        if self.enable_refiner_block_swap:
            # Create ledger WITHOUT LoRAs - loading will be fast
            stage_2_ledger_no_lora = ModelLedger(
                dtype=self.dtype,
                device=torch.device("cpu"),
                checkpoint_path=self._stage_2_checkpoint_path,
                gemma_root_path=self._stage_2_gemma_root,
                spatial_upsampler_path=self._stage_2_spatial_upsampler_path,
                vae_path=self._stage_2_vae_path,
                loras=(),  # No LoRAs - load base model only
                fp8transformer=self._stage_2_fp8transformer,
            )

            # Load transformer without LoRAs (fast - just loading weights)
            print(">>> Loading stage 2 transformer to CPU...", flush=True)
            transformer = stage_2_ledger_no_lora.transformer()

            # Now apply LoRAs using chunked GPU computation
            all_loras = (*self._stage_2_loras, *self._stage_2_distilled_lora) if self._stage_2_distilled_lora else self._stage_2_loras
            if all_loras:
                print(f">>> Loading {len(all_loras)} LoRA(s)...", flush=True)
                from ltx_core.loader.sft_loader import SafetensorsStateDictLoader
                lora_loader = SafetensorsStateDictLoader()
                lora_state_dicts = []
                lora_strengths = []
                for lora in all_loras:
                    lora_sd = lora_loader.load(lora.path, sd_ops=lora.sd_ops, device=torch.device("cpu"))
                    lora_state_dicts.append(lora_sd)
                    lora_strengths.append(lora.strength)

                # Apply LoRAs using chunked GPU computation
                apply_loras_chunked_gpu(
                    model=transformer,
                    lora_state_dicts=lora_state_dicts,
                    lora_strengths=lora_strengths,
                    gpu_device=self.device,
                    dtype=self.dtype,
                )

                # Clean up LoRA state dicts
                del lora_state_dicts
                synchronize_and_cleanup()

            # For VAE operations later
            stage_2_ledger = stage_2_ledger_no_lora

            # Move non-block components to GPU
            print(f">>> Enabling refiner block swapping ({self.refiner_blocks_in_memory} blocks in GPU)...")
            transformer.velocity_model.patchify_proj.to(self.device)
            transformer.velocity_model.adaln_single.to(self.device)
            transformer.velocity_model.caption_projection.to(self.device)
            transformer.velocity_model.norm_out.to(self.device)
            transformer.velocity_model.proj_out.to(self.device)
            transformer.velocity_model.scale_shift_table = torch.nn.Parameter(
                transformer.velocity_model.scale_shift_table.to(self.device)
            )
            if hasattr(transformer.velocity_model, "audio_patchify_proj"):
                transformer.velocity_model.audio_patchify_proj.to(self.device)
            if hasattr(transformer.velocity_model, "audio_adaln_single"):
                transformer.velocity_model.audio_adaln_single.to(self.device)
            if hasattr(transformer.velocity_model, "audio_caption_projection"):
                transformer.velocity_model.audio_caption_projection.to(self.device)
            if hasattr(transformer.velocity_model, "audio_norm_out"):
                transformer.velocity_model.audio_norm_out.to(self.device)
            if hasattr(transformer.velocity_model, "audio_proj_out"):
                transformer.velocity_model.audio_proj_out.to(self.device)
            if hasattr(transformer.velocity_model, "audio_scale_shift_table"):
                transformer.velocity_model.audio_scale_shift_table = torch.nn.Parameter(
                    transformer.velocity_model.audio_scale_shift_table.to(self.device)
                )
            # Cross-attention adaln components
            if hasattr(transformer.velocity_model, "av_ca_video_scale_shift_adaln_single"):
                transformer.velocity_model.av_ca_video_scale_shift_adaln_single.to(self.device)
            if hasattr(transformer.velocity_model, "av_ca_audio_scale_shift_adaln_single"):
                transformer.velocity_model.av_ca_audio_scale_shift_adaln_single.to(self.device)
            if hasattr(transformer.velocity_model, "av_ca_a2v_gate_adaln_single"):
                transformer.velocity_model.av_ca_a2v_gate_adaln_single.to(self.device)
            if hasattr(transformer.velocity_model, "av_ca_v2a_gate_adaln_single"):
                transformer.velocity_model.av_ca_v2a_gate_adaln_single.to(self.device)

            # Try to enable block swap with OOM retry (reduce blocks if needed)
            current_blocks = self.refiner_blocks_in_memory
            min_blocks = 1
            while current_blocks >= min_blocks:
                try:
                    # Use activation offload for extreme memory savings (moves activations to CPU between blocks)
                    if self.enable_activation_offload:
                        block_swap_manager = enable_block_swap_with_activation_offload(
                            transformer,
                            blocks_in_memory=current_blocks,
                            device=self.device,
                            verbose=True,
                            temporal_chunk_size=self.temporal_chunk_size,
                        )
                    else:
                        block_swap_manager = enable_block_swap(
                            transformer,
                            blocks_in_memory=current_blocks,
                            device=self.device,
                        )
                    # Update instance variable for later retry logic
                    self.refiner_blocks_in_memory = current_blocks
                    break
                except torch.OutOfMemoryError:
                    current_blocks -= 1
                    if current_blocks < min_blocks:
                        print(f">>> OOM Error during block swap setup: Already at minimum blocks ({min_blocks})")
                        raise
                    print(f">>> OOM during enable_block_swap! Retrying with {current_blocks} blocks...")
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    gc.collect()
        else:
            # Non-block-swap case: load with LoRAs directly to GPU (fast)
            stage_2_ledger = ModelLedger(
                dtype=self.dtype,
                device=self.device,
                checkpoint_path=self._stage_2_checkpoint_path,
                gemma_root_path=self._stage_2_gemma_root,
                spatial_upsampler_path=self._stage_2_spatial_upsampler_path,
                vae_path=self._stage_2_vae_path,
                loras=(*self._stage_2_loras, *self._stage_2_distilled_lora) if self._stage_2_distilled_lora else self._stage_2_loras,
                fp8transformer=self._stage_2_fp8transformer,
            )
            transformer = stage_2_ledger.transformer()

        # Enable FFN chunking for stage 2 transformer if configured
        if self.ffn_chunk_size is not None:
            set_ffn_chunk_size(transformer, self.ffn_chunk_size)

        # For refine-only mode, use the configurable refine_steps with strength scaling
        # For normal two-stage, use stage2_steps (default 3)
        if self.refine_only and input_video:
            # Generate base sigma schedule (1.0 to 0.0)
            base_sigmas = LTX2Scheduler().execute(steps=refine_steps).to(
                dtype=torch.float32, device=self.device
            )
            # Scale sigmas by refine_strength so they start from refine_strength instead of 1.0
            # This preserves (1 - refine_strength) of the input content
            distilled_sigmas = base_sigmas * refine_strength
            print(f">>> Using {refine_steps} refinement steps with strength {refine_strength}")
        else:
            # Use pre-tuned sigma values for stage 2 refinement with distilled LoRA
            # These values are specifically calibrated for the distilled model
            if stage2_steps == 3:
                # Use exact tuned values for default 3 steps
                distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)
            elif stage2_steps == 8:
                # Use full trained 8-step distilled schedule for better quality
                # This schedule was specifically trained and produces better results
                # than interpolating from the 3-step schedule
                distilled_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)
                print(f">>> Stage 2 using full trained 8-step distilled schedule")
            else:
                # Interpolate tuned values to support different step counts
                # Original tuned values: [0.909375, 0.725, 0.421875, 0.0]
                tuned = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES)
                # Create interpolation positions for original values (0 to 1)
                orig_positions = torch.linspace(0, 1, len(tuned))
                # Create new positions for desired step count
                new_positions = torch.linspace(0, 1, stage2_steps + 1)
                # Interpolate to get new sigma values
                distilled_sigmas = torch.zeros(stage2_steps + 1)
                for i, pos in enumerate(new_positions):
                    # Find surrounding original values and interpolate
                    idx = torch.searchsorted(orig_positions, pos, right=True)
                    if idx == 0:
                        distilled_sigmas[i] = tuned[0]
                    elif idx >= len(tuned):
                        distilled_sigmas[i] = tuned[-1]
                    else:
                        # Linear interpolation between tuned[idx-1] and tuned[idx]
                        t = (pos - orig_positions[idx-1]) / (orig_positions[idx] - orig_positions[idx-1])
                        distilled_sigmas[i] = tuned[idx-1] * (1 - t) + tuned[idx] * t
                distilled_sigmas = distilled_sigmas.to(self.device)
                print(f">>> Stage 2 using {stage2_steps} steps (interpolated from tuned schedule)")

        # Move text embeddings to GPU for stage 2 denoising
        v_context_p = v_context_p.to(self.device)
        a_context_p = a_context_p.to(self.device)

        # Define denoising function for stage 2 (no CFG, just positive)
        # Convert anchor_decay "none" to None for the denoising loop (if not already done in stage 1)
        effective_anchor_decay = anchor_decay if anchor_decay and anchor_decay != "none" else None
        def second_stage_denoising_loop(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: DiffusionStepProtocol,
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=v_context_p,
                    audio_context=a_context_p,
                    transformer=transformer,
                ),
                anchor_decay=effective_anchor_decay,
                callback=preview_callback,
                callback_interval=preview_callback_interval,
                latent_norm_fn=latent_norm_fn,
            )

        # Stage 2 output shape (full resolution)
        stage_2_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width,
            height=height,
            fps=frame_rate,
        )

        # Image conditioning for stage 2 (i2v - replaces latent at frame 0)
        # In refine-only mode, preserve image-conditioned frames without adding noise
        # by using strength=1.0 (which sets denoise_mask=0 for those frames)
        if self.refine_only and input_video and images:
            images_for_conditioning = [(path, idx, 1.0) for path, idx, _ in images]
        else:
            images_for_conditioning = images

        # Load video encoder for stage 2 image conditioning if needed
        stage_2_video_encoder = video_encoder
        if images_for_conditioning and stage_2_video_encoder is None:
            stage_2_video_encoder = self.stage_1_model_ledger.video_encoder()

        stage_2_conditionings = image_conditionings_by_replacing_latent(
            images=images_for_conditioning,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=stage_2_video_encoder,
            dtype=dtype,
            device=self.device,
        )

        # Anchor image conditioning for stage 2 (guiding latent - appends tokens)
        if anchor_interval is not None:
            anchor_tuples = build_anchor_image_tuples(
                anchor_image=anchor_image,
                anchor_interval=anchor_interval,
                anchor_strength=anchor_strength,
                num_frames=num_frames,
                images=images,
            )
            if anchor_tuples:
                # Load video encoder for anchor conditioning if needed
                if stage_2_video_encoder is None:
                    stage_2_video_encoder = self.stage_1_model_ledger.video_encoder()
                anchor_conditionings = image_conditionings_by_adding_guiding_latent(
                    images=anchor_tuples,
                    height=stage_2_output_shape.height,
                    width=stage_2_output_shape.width,
                    video_encoder=stage_2_video_encoder,
                    dtype=dtype,
                    device=self.device,
                )
                stage_2_conditionings = stage_2_conditionings + anchor_conditionings

        # Note: V2V video conditioning is NOT applied to stage 2 (matching ic_lora.py)
        # Stage 2 only refines the upscaled latent from stage 1 - video guidance already applied there

        # V2A Mode: Add frozen video conditioning for stage 2
        # Unlike V2V, V2A needs conditioning at stage 2 to keep video exactly frozen
        if v2a_mode and input_video:
            from ltx_core.conditioning import VideoConditionByLatentIndex
            print(f">>> V2A Stage 2: Encoding video at full resolution...")
            v2a_s2_start = time.time()

            # Load video encoder for stage 2 if needed
            if stage_2_video_encoder is None:
                stage_2_video_encoder = self.stage_1_model_ledger.video_encoder()

            # Load input video at stage 2 (full) resolution
            video_tensor = load_video_conditioning(
                video_path=input_video,
                height=stage_2_output_shape.height,
                width=stage_2_output_shape.width,
                frame_cap=num_frames,
                dtype=dtype,
                device=torch.device("cpu"),
            )

            actual_frames = video_tensor.shape[2]
            latent_stride = 8

            for frame_idx in range(0, actual_frames, latent_stride):
                latent_idx = frame_idx // latent_stride
                frame = video_tensor[:, :, frame_idx:frame_idx+1, :, :].to(device=self.device, dtype=dtype)

                with torch.no_grad():
                    encoded_frame = stage_2_video_encoder(frame)

                stage_2_conditionings.append(
                    VideoConditionByLatentIndex(
                        latent=encoded_frame.cpu(),
                        strength=1.0,
                        latent_idx=latent_idx,
                    )
                )
                del frame, encoded_frame
                torch.cuda.empty_cache()

            del video_tensor
            cleanup_memory()

            num_latent_frames = (actual_frames - 1) // latent_stride + 1
            print(f">>> V2A: {num_latent_frames} video frames frozen (stage 2) in {time.time() - v2a_s2_start:.1f}s")

        # SVI Pro: Add motion latent conditionings for stage 2
        if _motion_latent is not None and _num_motion_latent > 0:
            motion_cond = VideoConditionByMotionLatent(
                motion_latent=_motion_latent,
                strength=0.7,
                frame_offset=0,
            )
            motion_conditionings = motion_cond.to_keyframe_conditionings(
                num_frames=num_frames,
                dtype=dtype,
                device=self.device,
            )
            stage_2_conditionings = stage_2_conditionings + motion_conditionings

        # Depth Control: Add depth conditioning for stage 2 (optional)
        if depth_tensor is not None and depth_stage2:
            import torch.nn.functional as F
            print(">>> Depth Control: Adding depth conditioning to stage 2...")
            # Upscale depth to stage 2 resolution
            depth_tensor_s2 = F.interpolate(
                depth_tensor,
                size=(num_frames, stage_2_output_shape.height, stage_2_output_shape.width),
                mode='trilinear',
                align_corners=False,
            )
            # Load video encoder for stage 2 if needed
            if stage_2_video_encoder is None:
                stage_2_video_encoder = self.stage_1_model_ledger.video_encoder()
            depth_conditionings_s2 = encode_depth_conditioning(
                depth_tensor=depth_tensor_s2,
                video_encoder=stage_2_video_encoder,
                num_frames=num_frames,
                strength=depth_strength,
                device=self.device,
                dtype=dtype,
            )
            stage_2_conditionings = stage_2_conditionings + depth_conditionings_s2
            print(f">>> Depth Control: Added {len(depth_conditionings_s2)} depth conditioning chunks to stage 2")

        # Cleanup stage_2_video_encoder if it was loaded separately (not reusing video_encoder)
        if stage_2_video_encoder is not None and stage_2_video_encoder is not video_encoder:
            stage_2_video_encoder.to("cpu")
            del stage_2_video_encoder
            synchronize_and_cleanup()
        stage_2_video_encoder = None

        # Refine-only mode: Add keyframe conditionings from input video
        # This uses every 8th frame as conditioning with strength=1.0
        # to guide generation while avoiding chunked encoding glitches
        if refine_keyframe_conditionings:
            stage_2_conditionings = stage_2_conditionings + refine_keyframe_conditionings
            print(f">>> Added {len(refine_keyframe_conditionings)} keyframe conditionings from input video")

        # Note: Overlap latent conditioning is only applied in stage 1.
        # Stage 2 operates at full resolution while overlap latent is encoded at stage 1 resolution.
        # Stage 1 conditioning is sufficient for establishing coherence.

        print(f">>> Stage 2: Refining at {stage_2_output_shape.width}x{stage_2_output_shape.height}...")
        # For refine-only mode, use audio_latent from input video encoding
        # For v2v mode, use v2v_audio_latent if available
        # For normal generation, use audio_state.latent from stage 1
        if self.refine_only and input_video:
            # Get expected audio shape for Stage 2
            from ltx_core.types import AudioLatentShape
            expected_audio_shape = AudioLatentShape.from_video_pixel_shape(stage_2_output_shape)
            expected_shape = expected_audio_shape.to_torch_shape()

            if audio_latent is not None:
                actual_shape = audio_latent.shape
                if actual_shape != expected_shape:
                    print(f">>> [Refine Audio] Shape mismatch: actual {actual_shape} vs expected {expected_shape}")
                    # Pad or trim frames dimension (dim 2) to match expected
                    actual_frames = actual_shape[2]
                    expected_frames = expected_shape[2]
                    if actual_frames < expected_frames:
                        pad_frames = expected_frames - actual_frames
                        audio_latent = torch.nn.functional.pad(audio_latent, (0, 0, 0, pad_frames))
                        print(f">>> [Refine Audio] Padded audio latent by {pad_frames} frames")
                    elif actual_frames > expected_frames:
                        audio_latent = audio_latent[:, :, :expected_frames, :]
                        print(f">>> [Refine Audio] Trimmed audio latent by {actual_frames - expected_frames} frames")

            stage_2_initial_audio = audio_latent
        elif v2v_audio_latent is not None:
            # Get expected audio shape for Stage 2
            from ltx_core.types import AudioLatentShape
            expected_audio_shape = AudioLatentShape.from_video_pixel_shape(stage_2_output_shape)
            expected_shape = expected_audio_shape.to_torch_shape()
            actual_shape = v2v_audio_latent.shape

            print(">>> Using audio from input video")

            # Check if shapes match and fix if needed
            if actual_shape != expected_shape:
                print(f">>> [V2V Audio] Shape mismatch: actual {actual_shape} vs expected {expected_shape}")
                # Pad or trim frames dimension (dim 2) to match expected
                actual_frames = actual_shape[2]
                expected_frames = expected_shape[2]
                if actual_frames < expected_frames:
                    # Pad with zeros
                    pad_frames = expected_frames - actual_frames
                    v2v_audio_latent = torch.nn.functional.pad(v2v_audio_latent, (0, 0, 0, pad_frames))
                    print(f">>> [V2V Audio] Padded audio latent by {pad_frames} frames")
                elif actual_frames > expected_frames:
                    # Trim
                    v2v_audio_latent = v2v_audio_latent[:, :, :expected_frames, :]
                    print(f">>> [V2V Audio] Trimmed audio latent by {actual_frames - expected_frames} frames")

                # Check mel_bins dimension (dim 3) as well
                actual_mel = actual_shape[3]
                expected_mel = expected_shape[3]
                if actual_mel != expected_mel:
                    print(f">>> [V2V Audio] mel_bins mismatch: {actual_mel} vs {expected_mel}")
                    # This is a more serious issue - likely config mismatch
                    raise ValueError(f"Audio latent mel_bins mismatch: got {actual_mel}, expected {expected_mel}")

            stage_2_initial_audio = v2v_audio_latent
        else:
            stage_2_initial_audio = audio_state.latent

        # OOM retry loop - reduces blocks in GPU until denoising succeeds
        current_blocks = self.refiner_blocks_in_memory
        min_blocks = 1  # Minimum blocks to try before giving up
        max_retries = current_blocks - min_blocks + 1  # Maximum retry attempts

        for retry_attempt in range(max_retries):
            try:
                # Reset generator for reproducibility on retries
                if retry_attempt > 0:
                    generator = torch.Generator(device=self.device).manual_seed(seed)
                    noiser = GaussianNoiser(generator=generator)

                video_state, audio_state = denoise_audio_video(
                    output_shape=stage_2_output_shape,
                    conditionings=stage_2_conditionings,
                    noiser=noiser,
                    sigmas=distilled_sigmas,
                    stepper=stepper,
                    denoising_loop_fn=second_stage_denoising_loop,
                    components=self.pipeline_components,
                    dtype=dtype,
                    device=self.device,
                    noise_scale=distilled_sigmas[0],
                    initial_video_latent=upscaled_video_latent,
                    initial_audio_latent=stage_2_initial_audio,
                    audio_conditionings=audio_conditionings if audio_conditionings else None,
                )
                # Success - break out of retry loop
                break

            except torch.OutOfMemoryError as e:
                # Check if we can retry with fewer blocks
                if not block_swap_manager:
                    print(f">>> OOM Error: Block swapping not enabled, cannot reduce memory usage")
                    raise

                current_blocks -= 1
                if current_blocks < min_blocks:
                    print(f">>> OOM Error: Already at minimum blocks ({min_blocks}), cannot reduce further")
                    raise

                print(f">>> OOM Error caught! Retrying with {current_blocks} blocks in GPU...")

                # Clear CUDA error state and memory
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                gc.collect()

                # Reconfigure block swap with fewer blocks
                block_swap_manager, current_blocks = reconfigure_block_swap(
                    transformer,
                    new_blocks_in_memory=current_blocks,
                    device=torch.device(self.device),
                    enable_activation_offload=self.enable_activation_offload,
                    temporal_chunk_size=self.temporal_chunk_size,
                )

                if block_swap_manager is None:
                    print(f">>> Failed to reconfigure block swap, cannot retry")
                    raise

                # Clear memory again after reconfiguration
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                gc.collect()

                print(f">>> Retry attempt {retry_attempt + 1}/{max_retries}...")

        print(f">>> Stage 2 completed in {time.time() - stage2_start:.1f}s")

        # Clear closure references BEFORE transformer cleanup to break reference cycle
        # The denoising_loop closure captures transformer, preventing garbage collection
        second_stage_denoising_loop = None

        # Cleanup stage 2 models
        if block_swap_manager:
            offload_all_blocks(transformer)
            # Clear offloader references to break reference cycle
            if hasattr(transformer, 'velocity_model'):
                if hasattr(transformer.velocity_model, '_block_swap_offloader'):
                    transformer.velocity_model._block_swap_offloader = None
                if hasattr(transformer.velocity_model, '_blocks_ref'):
                    transformer.velocity_model._blocks_ref = None
            if hasattr(transformer, '_block_swap_offloader'):
                transformer._block_swap_offloader = None
            if hasattr(transformer, '_blocks_ref'):
                transformer._blocks_ref = None
            block_swap_manager = None

        # Move transformer to CPU before deletion to ensure GPU memory is freed
        if transformer is not None:
            transformer.to("cpu")
        transformer = None
        video_encoder = None

        # Clear stage 2 conditioning tensors to free GPU memory
        del stage_2_conditionings
        if 'upscaled_video_latent' in dir() and upscaled_video_latent is not None:
            del upscaled_video_latent

        # Move text embeddings to CPU (no longer needed on GPU)
        v_context_p = v_context_p.cpu()
        a_context_p = a_context_p.cpu()

        torch.cuda.synchronize()
        cleanup_memory()
        phase_barrier("stage_2_denoising")

        # =====================================================================
        # Phase 5: VAE Decoding (Sequential loading to minimize peak memory)
        # =====================================================================
        print(">>> Decoding video...")
        decode_start = time.time()

        # Load video decoder, decode, then offload to free memory before audio
        video_decoder = self.stage_2_model_ledger.video_decoder()
        # vae_decode_video returns a generator - consume it immediately before offloading decoder
        decoded_video_chunks = []
        for chunk in vae_decode_video(
            video_state.latent,
            video_decoder,
            tiling_config,
        ):
            decoded_video_chunks.append(chunk.cpu())  # Move chunks to CPU immediately
            del chunk
        # Create a generator that yields the cached chunks for encode_video compatibility
        def decoded_video_iter():
            for chunk in decoded_video_chunks:
                yield chunk.cuda()  # Move back to GPU when consumed
        decoded_video = decoded_video_iter()
        # Offload video decoder before loading audio models
        video_decoder.to("cpu")
        del video_decoder
        synchronize_and_cleanup()
        phase_barrier("video_decoding_two_stage")

        if not disable_audio:
            print(">>> Decoding audio...")
            # Load audio models only after video decoder is offloaded
            audio_decoder = self.stage_2_model_ledger.audio_decoder()
            vocoder = self.stage_2_model_ledger.vocoder()
            decoded_audio = vae_decode_audio(
                audio_state.latent,
                audio_decoder,
                vocoder,
            )
            # Offload audio models
            audio_decoder.to("cpu")
            vocoder.to("cpu")
            del audio_decoder, vocoder
            synchronize_and_cleanup()
        else:
            decoded_audio = None

        print(f">>> Decoding completed in {time.time() - decode_start:.1f}s")
        print(f">>> Total generation time: {time.time() - start_time:.1f}s")

        return decoded_video, decoded_audio, enhanced_prompt


# =============================================================================
# SVI Pro Multi-Clip Functions
# =============================================================================

def generate_svi_multi_clip(
    generator: "LTXVideoGeneratorWithOffloading",
    args,
    initial_image_path: str,
    num_clips: int,
    num_motion_latent: int = 2,
    num_motion_frame: int = 1,
    seed_multiplier: int = 42,
    overlap_frames: int = 1,
    prompts: list[str] | None = None,
    initial_motion_latent: torch.Tensor | None = None,
    latent_norm_fn: Callable | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Generate multi-clip streaming video using SVI Pro approach.

    This implements the SVI Pro algorithm adapted for LTX:
    1. Generate first clip from initial image (with anchor at frame 0)
    2. For each subsequent clip:
       - Extract motion frame from previous clip as new input image
       - Extract last N latent frames as motion_latent
       - Use original image as anchor for style consistency
       - Add motion_latent as keyframe conditionings at END of sequence
       - Vary seed per clip for different motion dynamics

    Args:
        generator: LTXVideoGeneratorWithOffloading instance
        args: Command line arguments containing generation parameters
        initial_image_path: Path to the starting image (becomes anchor)
        num_clips: Number of clips to generate
        num_motion_latent: Latent frames from previous clip for motion context
        num_motion_frame: Pixel frame offset from end for next input image
        seed_multiplier: Per-clip seed variation
        overlap_frames: Overlapping frames between clips
        prompts: Optional list of per-clip prompts
        initial_motion_latent: Optional motion latent for video extension

    Returns:
        Tuple of (concatenated_video_tensor [F,H,W,C], combined_audio_tensor or None)
    """
    import tempfile
    import shutil
    import cv2
    from PIL import Image

    device = generator.device
    dtype = generator.dtype
    base_seed = args.seed

    # Storage for clips
    all_video_chunks = []
    all_audio_chunks = []

    # SVI anchor: the original image for style consistency
    anchor_image_path = initial_image_path
    current_input_image = initial_image_path

    # Motion latent from previous clip (for clip 2+)
    prev_motion_latent = initial_motion_latent

    # Temp directory for intermediate frames
    temp_dir = tempfile.mkdtemp()

    try:
        for clip_idx in range(num_clips):
            print("=" * 60)
            print(f"SVI Pro: Generating clip {clip_idx + 1}/{num_clips}")
            print("=" * 60)

            # Calculate seed for this clip
            clip_seed = base_seed + clip_idx * seed_multiplier
            print(f">>> Clip {clip_idx + 1} seed: {clip_seed}")

            # Select prompt for this clip
            if prompts and clip_idx < len(prompts):
                clip_prompt = prompts[clip_idx]
            elif prompts:
                clip_prompt = prompts[-1]  # Use last prompt
            else:
                clip_prompt = args.prompt
            print(f">>> Clip {clip_idx + 1} prompt: {clip_prompt[:80]}..." if len(clip_prompt) > 80 else f">>> Clip {clip_idx + 1} prompt: {clip_prompt}")

            # Build image conditionings for this clip
            # Frame 0: current input image (motion frame from prev clip, or initial)
            clip_images = [(current_input_image, 0, 0.95)]

            # Create preview callback if enabled
            preview_callback = None
            if args.preview_dir:
                # Compute latent dimensions for preview unpatchification
                # Two-stage operates at half resolution in stage 1
                if generator.one_stage:
                    lf = (args.num_frames - 1) // 8 + 1
                    lh = args.height // 32
                    lw = args.width // 32
                else:
                    lf = (args.num_frames - 1) // 8 + 1
                    lh = (args.height // 2) // 32
                    lw = (args.width // 2) // 32

                preview_callback = create_preview_callback(
                    preview_dir=args.preview_dir,
                    preview_interval=args.preview_interval,
                    max_height=args.preview_max_height,
                    stage_name=f"svi_clip{clip_idx + 1}",
                    preview_suffix=args.preview_suffix,
                    latent_frames=lf,
                    latent_height=lh,
                    latent_width=lw,
                )

            # Generate this clip
            video_iterator, audio, _ = generator.generate(
                prompt=clip_prompt,
                negative_prompt=args.negative_prompt,
                seed=clip_seed,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                frame_rate=args.frame_rate,
                num_inference_steps=args.num_inference_steps,
                cfg_guidance_scale=args.cfg_guidance_scale,
                images=clip_images,
                tiling_config=TilingConfig.default(),
                enhance_prompt=False,
                disable_audio=args.disable_audio,
                stage2_steps=args.stage2_steps,
                anchor_image=anchor_image_path if clip_idx > 0 else None,
                anchor_interval=args.anchor_interval if clip_idx > 0 else None,
                anchor_strength=args.anchor_strength,
                anchor_decay=args.anchor_decay,
                audio=args.audio,
                audio_strength=args.audio_strength,
                # STG parameters
                stg_scale=args.stg_scale,
                stg_blocks=args.stg_blocks,
                stg_mode=args.stg_mode,
                # SVI-specific parameters
                _motion_latent=prev_motion_latent if clip_idx > 0 else None,
                _num_motion_latent=num_motion_latent if clip_idx > 0 else 0,
                preview_callback=preview_callback,
                preview_callback_interval=args.preview_interval,
                # Depth Control (IC-LoRA) parameters
                depth_video=args.depth_video,
                depth_image=args.depth_image,
                estimate_depth=args.estimate_depth,
                depth_strength=args.depth_strength,
                depth_stage2=args.depth_stage2,
                # Latent normalization
                latent_norm_fn=latent_norm_fn,
            )

            # Collect video frames from iterator - move to CPU immediately to save GPU memory
            video_frames = []
            for chunk in video_iterator:
                video_frames.append(chunk.cpu())  # Move to CPU immediately
                del chunk  # Free GPU memory
            # Concatenate on CPU (downstream ops handle device: _encode_frames_to_latent, .cpu().numpy())
            video_tensor = torch.cat(video_frames, dim=0)  # [F, H, W, C] on CPU
            print(f">>> Clip {clip_idx + 1} generated: {video_tensor.shape}")

            # Free video_frames list immediately - chunks are now in video_tensor
            del video_frames
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Extract motion latent for next clip
            if clip_idx < num_clips - 1 and num_motion_latent > 0:
                # We need frames for latent encoding
                # LTX temporal compression is 8x, so num_motion_latent * 8 pixel frames ≈ num_motion_latent latent frames
                frames_needed = num_motion_latent * 8 + 1  # Extra frame for VAE boundary
                frames_for_latent = video_tensor[-frames_needed:, :, :, :].clone()  # Clone to avoid view

                # Load video encoder for latent extraction
                video_encoder = generator.stage_1_model_ledger.video_encoder()
                new_motion_latent = _encode_frames_to_latent(
                    frames_for_latent,
                    video_encoder,
                    device,
                    dtype,
                    num_motion_latent,
                )
                print(f">>> Extracted motion latent: {new_motion_latent.shape}")

                # Cleanup encoder and intermediate tensors
                del frames_for_latent
                video_encoder.to("cpu")
                del video_encoder
                synchronize_and_cleanup()

                # Update motion latent (old one will be garbage collected)
                prev_motion_latent = new_motion_latent

                # Extract motion frame for next clip input
                frame_idx = -min(num_motion_frame, video_tensor.shape[0])
                motion_frame = video_tensor[frame_idx].numpy().astype("uint8")  # [H, W, C] - already on CPU

                temp_image_path = os.path.join(temp_dir, f"clip_{clip_idx}_motion.png")
                Image.fromarray(motion_frame).save(temp_image_path)
                current_input_image = temp_image_path
                print(f">>> Saved motion frame to: {temp_image_path}")

            # Store clip (skip overlap frames for non-first clips)
            # video_tensor is already on CPU from earlier collection
            if clip_idx == 0:
                all_video_chunks.append(video_tensor)
            else:
                all_video_chunks.append(video_tensor[overlap_frames:].clone())

            if audio is not None:
                # Calculate corresponding audio samples to skip
                # audio is at AUDIO_SAMPLE_RATE, video at args.frame_rate
                # Store audio on CPU as well
                if clip_idx == 0:
                    all_audio_chunks.append(audio.cpu() if audio.device.type != "cpu" else audio)
                else:
                    samples_per_frame = int(24000 / args.frame_rate)  # AUDIO_SAMPLE_RATE
                    samples_to_skip = overlap_frames * samples_per_frame
                    audio_chunk = audio[samples_to_skip:]
                    all_audio_chunks.append(audio_chunk.cpu() if audio_chunk.device.type != "cpu" else audio_chunk)

            # Cleanup GPU memory before next iteration
            del video_tensor
            if audio is not None:
                del audio
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()

        # Concatenate all clips
        print("=" * 60)
        print("SVI Pro: Concatenating clips...")
        print("=" * 60)
        final_video = torch.cat(all_video_chunks, dim=0)
        final_audio = torch.cat(all_audio_chunks, dim=0) if all_audio_chunks else None

        print(f">>> Final video shape: {final_video.shape}")
        if final_audio is not None:
            print(f">>> Final audio shape: {final_audio.shape}")

        return final_video, final_audio

    finally:
        # Cleanup temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def generate_svi_video_extension(
    generator: "LTXVideoGeneratorWithOffloading",
    args,
    input_video_path: str,
    num_clips: int,
    num_motion_latent: int = 2,
    num_motion_frame: int = 1,
    seed_multiplier: int = 42,
    overlap_frames: int = 1,
    anchor_image_path: str | None = None,
    prepend_original: bool = True,
    prompts: list[str] | None = None,
    latent_norm_fn: Callable | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Extend an existing video using SVI Pro approach.

    1. Load input video and extract transition frame (for next clip input)
    2. Extract and encode last N frames as initial motion latent
    3. Optionally extract first frame as anchor (or use provided)
    4. Call generate_svi_multi_clip with initial_motion_latent
    5. Concatenate original video with extension

    Args:
        generator: LTXVideoGeneratorWithOffloading instance
        args: Command line arguments
        input_video_path: Path to video to extend
        num_clips: Number of extension clips
        num_motion_latent: Motion latent frames
        num_motion_frame: Frame offset for input image
        seed_multiplier: Per-clip seed variation
        overlap_frames: Overlap for blending
        anchor_image_path: Optional explicit anchor image
        prepend_original: Whether to include original video
        prompts: Optional per-clip prompts

    Returns:
        Tuple of (extended_video, audio)
    """
    import tempfile
    import shutil
    import cv2
    from PIL import Image

    device = generator.device
    dtype = generator.dtype

    temp_dir = tempfile.mkdtemp()

    try:
        print("=" * 60)
        print("SVI Pro Video Extension Mode")
        print("=" * 60)

        # Load input video info
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f">>> Input video: {total_frames} frames at {fps:.1f} fps, {video_width}x{video_height}")

        # Extract transition frame (last frame for continuation)
        transition_frame_idx = total_frames - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, transition_frame_idx)
        ret, transition_frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {transition_frame_idx}")

        # Convert BGR to RGB
        transition_frame = cv2.cvtColor(transition_frame, cv2.COLOR_BGR2RGB)
        start_image_path = os.path.join(temp_dir, "transition_frame.png")
        Image.fromarray(transition_frame).save(start_image_path)
        print(f">>> Extracted transition frame to: {start_image_path}")

        # Extract anchor frame (first frame or provided)
        if anchor_image_path and os.path.exists(anchor_image_path):
            anchor_path = anchor_image_path
            print(f">>> Using provided anchor image: {anchor_path}")
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, first_frame = cap.read()
            if ret:
                first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
                anchor_path = os.path.join(temp_dir, "anchor_frame.png")
                Image.fromarray(first_frame).save(anchor_path)
                print(f">>> Extracted anchor frame to: {anchor_path}")
            else:
                anchor_path = start_image_path

        # Load original video frames for prepending
        original_video = None
        if prepend_original:
            print(">>> Loading original video frames...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            original_frames = []
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize to match target resolution
                if frame.shape[1] != args.width or frame.shape[0] != args.height:
                    frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_LANCZOS4)
                original_frames.append(torch.from_numpy(frame))
            original_video = torch.stack(original_frames, dim=0)  # [F, H, W, C]
            print(f">>> Loaded original video: {original_video.shape}")

        cap.release()

        # Encode last N frames for initial motion latent
        initial_motion_latent = None
        if num_motion_latent > 0 and original_video is not None:
            print(">>> Encoding initial motion latent from video end...")
            frames_needed = num_motion_latent * 8 + 1
            frames_for_latent = original_video[-frames_needed:, :, :, :].float()

            # Need to resize to stage 1 dimensions if using two-stage
            if not generator.one_stage:
                # Resize frames for stage 1
                stage_1_h = args.height // 2
                stage_1_w = args.width // 2
                frames_for_latent_resized = []
                for i in range(frames_for_latent.shape[0]):
                    frame = frames_for_latent[i].numpy().astype("uint8")
                    frame_resized = cv2.resize(frame, (stage_1_w, stage_1_h), interpolation=cv2.INTER_LANCZOS4)
                    frames_for_latent_resized.append(torch.from_numpy(frame_resized).float())
                frames_for_latent = torch.stack(frames_for_latent_resized, dim=0)

            video_encoder = generator.stage_1_model_ledger.video_encoder()
            initial_motion_latent = _encode_frames_to_latent(
                frames_for_latent,
                video_encoder,
                device,
                dtype,
                num_motion_latent,
            )
            print(f">>> Extracted initial motion latent: {initial_motion_latent.shape}")

            video_encoder.to("cpu")
            del video_encoder
            synchronize_and_cleanup()

        # Update anchor path in args for generate
        original_anchor = getattr(args, 'anchor_image', None)
        args.anchor_image = anchor_path

        # Generate extension clips
        extension_video, extension_audio = generate_svi_multi_clip(
            generator=generator,
            args=args,
            initial_image_path=start_image_path,
            num_clips=num_clips,
            num_motion_latent=num_motion_latent,
            num_motion_frame=num_motion_frame,
            seed_multiplier=seed_multiplier,
            overlap_frames=overlap_frames,
            prompts=prompts,
            initial_motion_latent=initial_motion_latent,
            latent_norm_fn=latent_norm_fn,
        )

        # Restore args
        args.anchor_image = original_anchor

        # Concatenate with original if prepending
        if prepend_original and original_video is not None:
            print(">>> Concatenating original video with extension...")
            # Blend overlap region for smooth transition
            if overlap_frames > 0 and extension_video.shape[0] > overlap_frames:
                # Create linear crossfade weights
                weights = torch.linspace(1.0, 0.0, overlap_frames, device=original_video.device)
                weights = weights.view(-1, 1, 1, 1)

                # Get overlap regions
                overlap_orig = original_video[-overlap_frames:].float()
                overlap_ext = extension_video[:overlap_frames].float()

                # Blend: original fades out, extension fades in
                blended = (overlap_orig * weights + overlap_ext * (1 - weights)).to(original_video.dtype)

                # Concatenate: orig[:-overlap] + blended + ext[overlap:]
                final_video = torch.cat([
                    original_video[:-overlap_frames],
                    blended,
                    extension_video[overlap_frames:],
                ], dim=0)
            else:
                final_video = torch.cat([original_video, extension_video], dim=0)

            print(f">>> Final extended video: {final_video.shape}")
        else:
            final_video = extension_video

        return final_video, extension_audio

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


# =============================================================================
# AV Extension Generation (Time-Based Audio-Video Masking)
# =============================================================================

def generate_av_extension(
    generator: "LTXVideoGeneratorWithOffloading",
    args,
    input_video_path: str,
    start_time: float | None = None,
    end_time: float | None = None,
    extend_steps: int = 8,
    terminal: float = 0.1,
    slope_len: int = 3,
    skip_stage2: bool = False,
    latent_norm_fn: Callable | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Extend a video using time-based audio-video masking.

    This implements the LTXVSetAudioVideoMaskByTime approach from ComfyUI-LTXVideo:
    1. Load and encode input video and audio to latent space
    2. Create noise masks based on time windows (preserve before start_time, generate after)
    3. Run masked denoising to generate new content while preserving original
    4. Decode and return the extended video

    Args:
        generator: LTXVideoGeneratorWithOffloading instance
        args: Command line arguments
        input_video_path: Path to video to extend
        start_time: Time (seconds) to start generating new content (default: end of video)
        end_time: Time (seconds) to stop generation (default: start_time + 5)
        extend_steps: Number of denoising steps for extension
        terminal: Terminal sigma for partial denoising (smaller = smoother continuation)
        slope_len: Transition length at mask boundaries
        skip_stage2: Whether to skip stage 2 refinement

    Returns:
        Tuple of (video_tensor [F, H, W, C], audio_tensor or None)
    """
    import cv2
    from dataclasses import replace as dataclass_replace
    from ltx_core.types import LatentState, VideoPixelShape

    device = generator.device
    dtype = generator.dtype

    print("=" * 60)
    print("AV Extension Mode (Time-Based Audio-Video Masking)")
    print("=" * 60)

    # =========================================================================
    # Step 0: Trim input video to 8n+1 frames if needed (VAE requirement)
    # =========================================================================
    cap_check = cv2.VideoCapture(input_video_path)
    if not cap_check.isOpened():
        raise RuntimeError(f"Failed to open video: {input_video_path}")

    raw_frames = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_check.release()

    n = max(1, (raw_frames - 1) // 8)
    valid_frames = 8 * n + 1

    if valid_frames < raw_frames:
        import tempfile
        import subprocess
        print(f">>> Trimming input video: {raw_frames} -> {valid_frames} frames (8*{n}+1) for VAE compatibility")

        # Create trimmed video using ffmpeg
        trimmed_path = tempfile.mktemp(suffix=".mp4")
        cmd = [
            "ffmpeg", "-y", "-i", input_video_path,
            "-frames:v", str(valid_frames),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
            "-c:a", "copy",
            trimmed_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        input_video_path = trimmed_path
        print(f">>> Trimmed video saved to temp file")

    # =========================================================================
    # Step 1: Load and analyze input video
    # =========================================================================
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_duration = total_frames / input_fps

    print(f">>> Input video: {total_frames} frames at {input_fps:.1f} fps, {video_width}x{video_height}")
    print(f">>> Duration: {input_duration:.2f} seconds")

    # Calculate time parameters
    if start_time is None:
        start_time = input_duration  # Start generating at end of video
    if end_time is None:
        end_time = start_time + 5.0  # Generate 5 seconds by default

    print(f">>> Preserve: 0 - {start_time:.2f}s, Generate: {start_time:.2f} - {end_time:.2f}s")

    # Calculate total output frames
    output_fps = args.frame_rate
    output_frames = int(round(end_time * output_fps))
    # Ensure output frames is valid (8n+1 format)
    output_frames = ((output_frames - 1) // 8) * 8 + 1

    print(f">>> Output: {output_frames} frames at {output_fps} fps ({output_frames / output_fps:.2f}s)")

    # =========================================================================
    # Step 2: Load video frames and resize to target resolution
    # =========================================================================
    print(">>> Loading and resizing video frames...")

    # Determine output resolution
    out_width = args.width
    out_height = args.height

    # Use half resolution for two-stage pipeline
    if not generator.one_stage:
        stage1_width = out_width // 2
        stage1_height = out_height // 2
    else:
        stage1_width = out_width
        stage1_height = out_height

    # Ensure dimensions are divisible by 32
    stage1_width = (stage1_width // 32) * 32
    stage1_height = (stage1_height // 32) * 32

    # Load frames from input video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    input_frames = []
    for _ in range(min(total_frames, int(start_time * input_fps) + 16)):  # Load up to start_time + some buffer
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize to target resolution
        frame = cv2.resize(frame, (stage1_width, stage1_height), interpolation=cv2.INTER_LANCZOS4)
        input_frames.append(frame)
    cap.release()

    print(f">>> Loaded {len(input_frames)} frames, resized to {stage1_width}x{stage1_height}")

    # Convert to tensor [F, H, W, C]
    import numpy as np
    input_frames_tensor = torch.from_numpy(np.stack(input_frames)).float() / 255.0
    input_frames_tensor = input_frames_tensor.to(device=device, dtype=dtype)

    # =========================================================================
    # Step 3: Load audio from input video
    # =========================================================================
    print(">>> Extracting audio from input video...")
    audio_waveform = None
    audio_sample_rate = None

    try:
        # Pass device (not sample_rate!) to decode_audio_from_file
        # The function gets sample_rate from the audio stream itself
        waveform, sample_rate = decode_audio_from_file(input_video_path, device)
        if waveform is not None:
            audio_waveform = waveform
            audio_sample_rate = sample_rate
            print(f">>> Audio extracted: {waveform.shape}, sample rate: {sample_rate}")
        else:
            print(">>> No audio found in input video")
    except Exception as e:
        print(f">>> Failed to extract audio: {e}")

    # =========================================================================
    # Step 4: Encode video and audio to latent space
    # =========================================================================
    print(">>> Encoding video to latent space...")

    video_encoder = generator.stage_1_model_ledger.video_encoder()

    # Get the encoder's dtype (usually bfloat16)
    encoder_dtype = next(video_encoder.parameters()).dtype

    # Get VAE temporal scale factor
    time_scale_factor = video_encoder.downscale_index_formula[0] if hasattr(video_encoder, 'downscale_index_formula') else 8

    # Convert from [F, H, W, C] to [1, C, F, H, W] and normalize to [-1, 1]
    video_input = input_frames_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, F, H, W]
    video_input = video_input * 2.0 - 1.0  # Normalize to [-1, 1]

    # Encode in temporal chunks to avoid OOM
    # Each chunk should be 8*k+1 frames to align with latent structure
    chunk_pixel_frames = 65  # 8*8+1 = 65 frames per chunk (gives 9 latent frames)
    total_pixel_frames = video_input.shape[2]

    latent_chunks = []
    chunk_idx = 0

    with torch.no_grad():
        for start_frame in range(0, total_pixel_frames, chunk_pixel_frames - 1):  # Overlap by 1 frame for continuity
            end_frame = min(start_frame + chunk_pixel_frames, total_pixel_frames)
            actual_frames = end_frame - start_frame

            # Ensure we have at least 9 frames (minimum for VAE)
            if actual_frames < 9:
                # Pad with last frame
                pad_frames = 9 - actual_frames
                chunk = video_input[:, :, start_frame:end_frame, :, :]
                last_frame = chunk[:, :, -1:, :, :].expand(-1, -1, pad_frames, -1, -1)
                chunk = torch.cat([chunk, last_frame], dim=2)
            else:
                chunk = video_input[:, :, start_frame:end_frame, :, :]

            # Encode chunk
            chunk_latent = video_encoder(chunk.to(device=device, dtype=encoder_dtype))
            chunk_latent = chunk_latent.to(dtype=dtype)

            # For overlapping chunks, skip the first latent frame (except for first chunk)
            if chunk_idx > 0 and len(latent_chunks) > 0:
                chunk_latent = chunk_latent[:, :, 1:, :, :]

            latent_chunks.append(chunk_latent)
            chunk_idx += 1

            print(f">>> Encoded chunk {chunk_idx}: frames {start_frame}-{end_frame} -> {chunk_latent.shape[2]} latent frames")

            # Clear cache between chunks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Concatenate all chunks
    video_latent = torch.cat(latent_chunks, dim=2)
    print(f">>> Video latent shape: {video_latent.shape}")

    # FREE: Delete chunk list and input tensors no longer needed
    del latent_chunks
    del video_input
    del input_frames_tensor
    cleanup_memory()

    # Encode audio
    audio_latent = None
    audio_mel_hop_length = None
    if audio_waveform is not None and not args.disable_audio:
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

        # Convert to mel spectrogram
        mel_spectrogram = audio_processor.waveform_to_mel(
            audio_waveform.to(dtype=torch.float32),
            waveform_sample_rate=audio_sample_rate or AUDIO_SAMPLE_RATE
        )

        # Encode to latent
        with torch.no_grad():
            audio_latent = audio_encoder(mel_spectrogram.to(dtype=torch.float32).to(device))
            audio_latent = audio_latent.to(dtype=dtype)

        audio_mel_hop_length = audio_processor.mel_hop_length if hasattr(audio_processor, 'mel_hop_length') else 320

        print(f">>> Audio latent shape: {audio_latent.shape}")

        del audio_encoder, audio_processor
        cleanup_memory()

    del video_encoder
    cleanup_memory()

    # =========================================================================
    # Step 5: Create empty latents for extended video
    # =========================================================================
    print(">>> Creating extended latent space...")

    # Calculate required latent frames for output
    output_latent_frames = (output_frames - 1) // time_scale_factor + 1

    # Create extended video latent (copy original + add space for new content)
    input_latent_frames = video_latent.shape[2]
    extended_video_latent = torch.zeros(
        video_latent.shape[0],  # batch
        video_latent.shape[1],  # channels
        output_latent_frames,   # frames
        video_latent.shape[3],  # height
        video_latent.shape[4],  # width
        device=device,
        dtype=dtype,
    )

    # Copy original latent into the extended latent
    copy_frames = min(input_latent_frames, output_latent_frames)
    extended_video_latent[:, :, :copy_frames, :, :] = video_latent[:, :, :copy_frames, :, :]

    print(f">>> Extended video latent: {extended_video_latent.shape}")

    extended_audio_latent = None
    audio_latents_per_second = None
    if audio_latent is not None:
        input_audio_latent_frames = audio_latent.shape[2]
        input_audio_duration = start_time
        audio_latents_per_second = input_audio_latent_frames / input_audio_duration
        output_audio_latent_frames = int(round(end_time * audio_latents_per_second))

        extended_audio_latent = torch.zeros(
            audio_latent.shape[0],
            audio_latent.shape[1],
            output_audio_latent_frames,
            audio_latent.shape[3],
            device=device,
            dtype=dtype,
        )

        input_audio_frames = audio_latent.shape[2]
        copy_audio_frames = min(input_audio_frames, output_audio_latent_frames)
        extended_audio_latent[:, :, :copy_audio_frames, :] = audio_latent[:, :, :copy_audio_frames, :]

        print(f">>> Extended audio latent: {extended_audio_latent.shape}")

    # FREE: Delete original latents (already copied to extended versions)
    del video_latent
    if audio_latent is not None:
        del audio_latent
    cleanup_memory()

    # =========================================================================
    # Step 6: Create noise masks
    # =========================================================================
    print(">>> Creating noise masks...")

    video_mask, audio_mask = create_av_noise_mask(
        video_latent=extended_video_latent,
        audio_latent=extended_audio_latent,
        start_time=start_time,
        end_time=end_time,
        video_fps=output_fps,
        time_scale_factor=time_scale_factor,
        sampling_rate=AUDIO_SAMPLE_RATE if extended_audio_latent is not None else None,
        mel_hop_length=audio_mel_hop_length,
        init_video_mask=0.0,
        init_audio_mask=0.0,
        slope_len=slope_len,
        audio_latents_per_second=audio_latents_per_second if extended_audio_latent is not None else None,
    )

    # OFFLOAD: Move latents and masks to CPU before loading text encoder
    # This frees GPU memory for the large text encoder model
    print(">>> Offloading latents to CPU for model loading...")
    extended_video_latent = extended_video_latent.cpu()
    if extended_audio_latent is not None:
        extended_audio_latent = extended_audio_latent.cpu()
    video_mask = video_mask.cpu()
    if audio_mask is not None:
        audio_mask = audio_mask.cpu()
    cleanup_memory()

    # =========================================================================
    # Step 7: Run masked denoising
    # =========================================================================
    print(">>> Running masked denoising...")

    # 7A: Load text encoder FIRST and encode prompts (before transformer!)
    text_encoder = generator.stage_1_model_ledger.text_encoder()
    print(">>> Encoding prompts...")
    if args.cfg_guidance_scale > 1.0:
        context_p, context_n = encode_text(text_encoder, prompts=[args.prompt, args.negative_prompt])
        v_context_p, a_context_p = context_p
        v_context_n, a_context_n = context_n
    else:
        context_p = encode_text(text_encoder, prompts=[args.prompt])[0]
        v_context_p, a_context_p = context_p
        v_context_n, a_context_n = None, None

    # 7B: Delete text encoder BEFORE loading transformer
    del text_encoder
    cleanup_memory()

    # 7C: Reload latents to GPU (after text encoder is gone)
    print(">>> Reloading latents to GPU for denoising...")
    extended_video_latent = extended_video_latent.to(device=device, dtype=dtype)
    if extended_audio_latent is not None:
        extended_audio_latent = extended_audio_latent.to(device=device, dtype=dtype)
    video_mask = video_mask.to(device=device)
    if audio_mask is not None:
        audio_mask = audio_mask.to(device=device)

    # 7D: NOW load transformer (text encoder memory is freed)
    # Use the same block swapping pattern as main generate() if enabled
    block_swap_manager = None
    if generator.enable_dit_block_swap:
        print(f">>> Loading DiT transformer with block swapping ({generator.dit_blocks_in_memory} blocks in GPU)...", flush=True)
        from ltx_core.loader.sft_loader import SafetensorsStateDictLoader

        # Check if there are LoRAs to apply
        has_loras = hasattr(generator.stage_1_model_ledger, 'loras') and generator.stage_1_model_ledger.loras

        if has_loras:
            # Create ledger without LoRAs for fast base model loading
            stage_1_ledger_no_lora = ModelLedger(
                dtype=generator.dtype,
                device=torch.device("cpu"),
                checkpoint_path=generator.stage_1_model_ledger.checkpoint_path,
                gemma_root_path=generator.stage_1_model_ledger.gemma_root_path,
                spatial_upsampler_path=generator.stage_1_model_ledger.spatial_upsampler_path,
                vae_path=generator.stage_1_model_ledger.vae_path,
                loras=(),  # No LoRAs - load base model only
                fp8transformer=generator.stage_1_model_ledger.fp8transformer,
            )
            transformer = stage_1_ledger_no_lora.transformer()

            # Apply LoRAs using chunked GPU computation
            loras = generator.stage_1_model_ledger.loras
            print(f">>> Applying {len(loras)} LoRA(s) using chunked GPU computation...", flush=True)
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
            # No LoRAs - load directly to CPU
            original_device = generator.stage_1_model_ledger.device
            generator.stage_1_model_ledger.device = torch.device("cpu")
            transformer = generator.stage_1_model_ledger.transformer()
            generator.stage_1_model_ledger.device = original_device

        # Move non-block components to GPU, keep blocks on CPU
        transformer.velocity_model.patchify_proj.to(device)
        transformer.velocity_model.adaln_single.to(device)
        transformer.velocity_model.caption_projection.to(device)
        transformer.velocity_model.norm_out.to(device)
        transformer.velocity_model.proj_out.to(device)
        transformer.velocity_model.scale_shift_table = torch.nn.Parameter(
            transformer.velocity_model.scale_shift_table.to(device)
        )
        # Audio components
        if hasattr(transformer.velocity_model, "audio_patchify_proj"):
            transformer.velocity_model.audio_patchify_proj.to(device)
        if hasattr(transformer.velocity_model, "audio_adaln_single"):
            transformer.velocity_model.audio_adaln_single.to(device)
        if hasattr(transformer.velocity_model, "audio_caption_projection"):
            transformer.velocity_model.audio_caption_projection.to(device)
        if hasattr(transformer.velocity_model, "audio_norm_out"):
            transformer.velocity_model.audio_norm_out.to(device)
        if hasattr(transformer.velocity_model, "audio_proj_out"):
            transformer.velocity_model.audio_proj_out.to(device)
        if hasattr(transformer.velocity_model, "audio_scale_shift_table"):
            transformer.velocity_model.audio_scale_shift_table = torch.nn.Parameter(
                transformer.velocity_model.audio_scale_shift_table.to(device)
            )
        # Cross-attention components
        if hasattr(transformer.velocity_model, "av_ca_video_scale_shift_adaln_single"):
            transformer.velocity_model.av_ca_video_scale_shift_adaln_single.to(device)
        if hasattr(transformer.velocity_model, "av_ca_audio_scale_shift_adaln_single"):
            transformer.velocity_model.av_ca_audio_scale_shift_adaln_single.to(device)
        if hasattr(transformer.velocity_model, "av_ca_a2v_gate_adaln_single"):
            transformer.velocity_model.av_ca_a2v_gate_adaln_single.to(device)
        if hasattr(transformer.velocity_model, "av_ca_v2a_gate_adaln_single"):
            transformer.velocity_model.av_ca_v2a_gate_adaln_single.to(device)

        # Use activation offload for extreme memory savings (moves activations to CPU between blocks)
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

    sigmas = LTX2Scheduler().execute(
        steps=extend_steps,
        latent=extended_video_latent,
        terminal=terminal,
        stretch=True,
    ).to(dtype=torch.float32, device=device)

    # Initialize diffusion components
    generator_torch = torch.Generator(device=device).manual_seed(args.seed)
    noiser = GaussianNoiser(generator=generator_torch)
    stepper = EulerDiffusionStep()
    cfg_guider = CFGGuider(args.cfg_guidance_scale)
    # Initialize STG (Spatio-Temporal Guidance) components
    effective_stg_blocks = args.stg_blocks if args.stg_blocks is not None else [29]
    stg_guider = STGGuider(args.stg_scale)
    stg_perturbation_config = build_stg_perturbation_config(
        stg_scale=args.stg_scale,
        stg_blocks=effective_stg_blocks,
        stg_mode=args.stg_mode,
    )

    # Create output shape
    output_shape = VideoPixelShape(
        batch=1,
        frames=output_frames,
        height=stage1_height,
        width=stage1_width,
        fps=output_fps,
    )

    # Use the generator's pipeline components
    components = generator.pipeline_components

    # Run denoising loop
    print(f">>> Denoising with {len(sigmas) - 1} steps...")

    # Create the denoise function (CFG + STG combined)
    use_cfg = args.cfg_guidance_scale > 1.0 and v_context_n is not None
    use_stg = stg_guider.enabled() and stg_perturbation_config is not None
    if use_cfg or use_stg:
        denoise_fn = cfg_stg_denoising_func(
            cfg_guider=cfg_guider,
            stg_guider=stg_guider,
            stg_perturbation_config=stg_perturbation_config,
            v_context_p=v_context_p,
            v_context_n=v_context_n,
            a_context_p=a_context_p,
            a_context_n=a_context_n,
            transformer=transformer,
        )
    else:
        denoise_fn = simple_denoising_func(v_context_p, a_context_p, transformer)

    from ltx_core.types import VideoLatentShape, AudioLatentShape
    from ltx_core.tools import VideoLatentTools, AudioLatentTools
    from ltx_pipelines.utils.helpers import euler_denoising_loop
    from ltx_core.components.patchifiers import get_pixel_coords

    # =========================================================================
    # V2V-STYLE EXTENSION: Separate reference (preserved) and target (generated)
    # This matches the training regime where reference tokens are prepended
    # with their own positions, rather than sharing positions with mixed timesteps.
    # =========================================================================

    # Calculate the split point in latent frames
    video_start_idx = time_to_video_latent_idx(
        start_time, output_fps, time_scale_factor, extended_video_latent.shape[2]
    )

    # Split the extended latent into reference (preserved) and target (generated)
    # Reference: frames [0, video_start_idx) - the preserved input video
    # Target: frames [video_start_idx, total) - the frames to generate
    reference_latent = extended_video_latent[:, :, :video_start_idx, :, :]
    target_latent = extended_video_latent[:, :, video_start_idx:, :, :]

    print(f">>> V2V-style extension: {reference_latent.shape[2]} reference frames + {target_latent.shape[2]} target frames")

    # Create VideoLatentTools for reference and target separately
    ref_pixel_frames = (reference_latent.shape[2] - 1) * time_scale_factor + 1
    target_pixel_frames = (target_latent.shape[2] - 1) * time_scale_factor + 1

    ref_shape = VideoPixelShape(
        batch=1, frames=ref_pixel_frames,
        height=stage1_height, width=stage1_width, fps=output_fps,
    )
    ref_latent_shape = VideoLatentShape.from_pixel_shape(
        shape=ref_shape,
        latent_channels=components.video_latent_channels,
        scale_factors=components.video_scale_factors,
    )
    ref_tools = VideoLatentTools(
        patchifier=components.video_patchifier,
        target_shape=ref_latent_shape,
        fps=output_fps,
    )

    target_shape_ext = VideoPixelShape(
        batch=1, frames=target_pixel_frames,
        height=stage1_height, width=stage1_width, fps=output_fps,
    )
    target_latent_shape = VideoLatentShape.from_pixel_shape(
        shape=target_shape_ext,
        latent_channels=components.video_latent_channels,
        scale_factors=components.video_scale_factors,
    )
    target_tools = VideoLatentTools(
        patchifier=components.video_patchifier,
        target_shape=target_latent_shape,
        fps=output_fps,
    )

    # Create reference state (preserved frames, timestep=0, no noise)
    ref_state = ref_tools.create_initial_state(device, dtype, reference_latent)
    # Set mask to 0 for reference (no denoising - they're conditioning tokens)
    ref_state = dataclass_replace(
        ref_state,
        denoise_mask=torch.zeros_like(ref_state.denoise_mask),
    )
    # Reference doesn't get noise (mask=0 means clean latent)

    # Create target state (generated frames, timestep=sigma, full noise)
    target_state = target_tools.create_initial_state(device, dtype, target_latent)
    # Set mask to 1 for target (full denoising)
    target_state = dataclass_replace(
        target_state,
        denoise_mask=torch.ones_like(target_state.denoise_mask),
    )
    # Add noise to target
    target_state = noiser(target_state, noise_scale=1.0)

    # Offset target positions by the reference duration for temporal continuity
    # Target positions should start after the reference video ends
    target_positions = target_state.positions.clone()
    # Position tensor shape is [B, 3, seq_len, 2] where dim 1 is (time, height, width)
    # and dim 3 is (start, end) bounds. We offset the time dimension.
    # Note: ref_pixel_frames is in PIXEL frames, so dividing by fps gives seconds
    time_offset = ref_pixel_frames / output_fps  # Reference duration in seconds
    target_positions[:, 0, :, :] += time_offset
    print(f">>> Target position offset: {time_offset:.3f}s (ref has {ref_pixel_frames} pixel frames)")

    target_state = dataclass_replace(target_state, positions=target_positions)

    # Concatenate reference and target: [reference_tokens, target_tokens]
    # This matches v2v training order where reference comes first
    video_state = LatentState(
        latent=torch.cat([ref_state.latent, target_state.latent], dim=1),
        denoise_mask=torch.cat([ref_state.denoise_mask, target_state.denoise_mask], dim=1),
        positions=torch.cat([ref_state.positions, target_state.positions], dim=2),
        clean_latent=torch.cat([ref_state.clean_latent, target_state.clean_latent], dim=1),
    )

    # Store reference sequence length for extraction after denoising
    ref_seq_len = ref_state.latent.shape[1]

    # Also create tools for the full combined latent (for unpatchifying later)
    video_latent_shape = VideoLatentShape.from_pixel_shape(
        shape=output_shape,
        latent_channels=components.video_latent_channels,
        scale_factors=components.video_scale_factors,
    )
    video_tools = VideoLatentTools(
        patchifier=components.video_patchifier,
        target_shape=video_latent_shape,
        fps=output_fps,
    )

    if extended_audio_latent is not None:
        audio_latent_shape = AudioLatentShape.from_torch_shape(extended_audio_latent.shape)
        audio_tools = AudioLatentTools(
            patchifier=components.audio_patchifier,
            target_shape=audio_latent_shape,
        )
        audio_state = audio_tools.create_initial_state(device, dtype, extended_audio_latent)
        patchified_audio_mask = audio_tools.patchifier.patchify(audio_mask.to(device=device))
        audio_state = dataclass_replace(
            audio_state,
            denoise_mask=patchified_audio_mask.to(dtype=torch.float32),
        )
        audio_state = noiser(audio_state, noise_scale=1.0)
    else:
        audio_latent_shape = AudioLatentShape.from_video_pixel_shape(shape=output_shape)
        audio_tools = AudioLatentTools(
            patchifier=components.audio_patchifier,
            target_shape=audio_latent_shape,
        )
        audio_state = audio_tools.create_initial_state(device, dtype, None)
        audio_state = dataclass_replace(
            audio_state,
            denoise_mask=torch.zeros_like(audio_state.denoise_mask),
        )
        audio_state = noiser(audio_state, noise_scale=1.0)

    # Run denoising loop with properly masked states
    final_video_state, final_audio_state = euler_denoising_loop(
        sigmas=sigmas,
        video_state=video_state,
        audio_state=audio_state,
        stepper=stepper,
        denoise_fn=denoise_fn,
        latent_norm_fn=latent_norm_fn,
    )

    # =========================================================================
    # V2V-STYLE POST-PROCESSING: Extract target tokens and reconstruct full video
    # =========================================================================
    # After denoising, final_video_state has [ref_tokens, target_tokens] concatenated.
    # We only need the target tokens (the generated frames).
    # Reference tokens were at timestep=0 throughout, so they're essentially unchanged.
    # We reconstruct the full video by concatenating original preserved frames with generated.

    print(f">>> Extracting target tokens (skip first {ref_seq_len} reference tokens)...")

    # Extract only target portion from final_video_state
    target_final_state = LatentState(
        latent=final_video_state.latent[:, ref_seq_len:, :],
        denoise_mask=final_video_state.denoise_mask[:, ref_seq_len:, :],
        positions=final_video_state.positions[:, :, ref_seq_len:, :],
        clean_latent=final_video_state.clean_latent[:, ref_seq_len:, :] if final_video_state.clean_latent is not None else None,
    )

    # Clear conditioning and unpatchify only the target
    target_final_state = target_tools.clear_conditioning(target_final_state)
    target_final_state = target_tools.unpatchify(target_final_state)

    # Get the original preserved video (reference) in 5D latent format
    # This is the clean input video, not the model output
    reference_video_latent = extended_video_latent[:, :, :video_start_idx, :, :]

    # Get the generated target video in 5D latent format
    generated_video_latent = target_final_state.latent

    print(f">>> Reference latent: {reference_video_latent.shape}, Generated latent: {generated_video_latent.shape}")

    # Concatenate to get full video: [preserved_frames, generated_frames]
    denoised_video_latent = torch.cat([reference_video_latent, generated_video_latent], dim=2)
    print(f">>> Full denoised video latent: {denoised_video_latent.shape}")

    # Handle audio (still using original mask-based approach)
    if extended_audio_latent is not None:
        final_audio_state = audio_tools.clear_conditioning(final_audio_state)
        final_audio_state = audio_tools.unpatchify(final_audio_state)

    # =========================================================================
    # Step 8: Get denoised latents
    # =========================================================================
    print(">>> Denoising complete...")

    denoised_audio_latent = None
    if extended_audio_latent is not None:
        denoised_audio_latent = final_audio_state.latent

    # Delete stage 1 transformer with proper block swap cleanup
    if block_swap_manager is not None:
        offload_all_blocks(transformer)
        transformer.velocity_model._block_swap_offloader = None
        transformer.velocity_model._blocks_ref = None
        block_swap_manager = None
    del transformer
    cleanup_memory()
    phase_barrier("stage_1_denoising")

    # =========================================================================
    # Step 9: Stage 2 Refinement (if enabled) - operates on LATENTS, not pixels
    # =========================================================================
    if not skip_stage2 and not generator.one_stage:
        print(">>> Stage 2 refinement...")

        # 9A: Spatial upsampling of LATENT (2x in H/W)
        from ltx_core.model.upsampler import upsample_video as upsample_latent_fn

        video_encoder = generator.stage_1_model_ledger.video_encoder()
        spatial_upsampler = generator.stage_2_model_ledger.spatial_upsampler()

        print(">>> Upsampling latents (2x)...")
        upscaled_video_latent = upsample_latent_fn(
            latent=denoised_video_latent,
            video_encoder=video_encoder,
            upsampler=spatial_upsampler,
        )

        del spatial_upsampler
        cleanup_memory()

        # =====================================================================
        # 9A-FIX: Re-encode original video at full resolution for preserved frames
        # =====================================================================
        # The LatentUpsampler is a neural network that modifies ALL latent values,
        # corrupting the preserved frames. We fix this by:
        # 1. Re-loading the input video at full (Stage 2) resolution
        # 2. Encoding it to latent space
        # 3. Replacing the upsampler output's preserved frames with the original encoding
        print(">>> Loading original video at full resolution for preserved frames...")

        stage2_width = stage1_width * 2
        stage2_height = stage1_height * 2

        cap_full = cv2.VideoCapture(input_video_path)
        cap_full.set(cv2.CAP_PROP_POS_FRAMES, 0)
        full_res_frames = []
        frames_to_load = min(int(cap_full.get(cv2.CAP_PROP_FRAME_COUNT)), int(start_time * input_fps) + 16)
        for _ in range(frames_to_load):
            ret, frame = cap_full.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (stage2_width, stage2_height), interpolation=cv2.INTER_LANCZOS4)
            full_res_frames.append(frame)
        cap_full.release()

        print(f">>> Loaded {len(full_res_frames)} frames at {stage2_width}x{stage2_height}")

        # Convert to tensor and encode
        import numpy as np
        full_res_tensor = torch.from_numpy(np.stack(full_res_frames)).float() / 255.0
        full_res_tensor = full_res_tensor.to(device=device, dtype=dtype)
        full_res_input = full_res_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, F, H, W]
        full_res_input = full_res_input * 2.0 - 1.0  # Normalize to [-1, 1]

        # Encode in temporal chunks (same logic as Stage 1)
        encoder_dtype = next(video_encoder.parameters()).dtype
        chunk_pixel_frames = 65
        total_pixel_frames_full = full_res_input.shape[2]

        full_res_latent_chunks = []
        chunk_idx_full = 0

        print(">>> Encoding full-resolution video to latent space...")
        with torch.no_grad():
            for start_frame_idx in range(0, total_pixel_frames_full, chunk_pixel_frames - 1):
                end_frame_idx = min(start_frame_idx + chunk_pixel_frames, total_pixel_frames_full)
                actual_frames_chunk = end_frame_idx - start_frame_idx

                if actual_frames_chunk < 9:
                    pad_frames = 9 - actual_frames_chunk
                    chunk = full_res_input[:, :, start_frame_idx:end_frame_idx, :, :]
                    last_frame = chunk[:, :, -1:, :, :].expand(-1, -1, pad_frames, -1, -1)
                    chunk = torch.cat([chunk, last_frame], dim=2)
                else:
                    chunk = full_res_input[:, :, start_frame_idx:end_frame_idx, :, :]

                chunk_latent = video_encoder(chunk.to(device=device, dtype=encoder_dtype))
                chunk_latent = chunk_latent.to(dtype=dtype)

                if chunk_idx_full > 0 and len(full_res_latent_chunks) > 0:
                    chunk_latent = chunk_latent[:, :, 1:, :, :]

                full_res_latent_chunks.append(chunk_latent)
                chunk_idx_full += 1

                print(f">>> Encoded full-res chunk {chunk_idx_full}: frames {start_frame_idx}-{end_frame_idx} -> {chunk_latent.shape[2]} latent frames")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        full_res_latent = torch.cat(full_res_latent_chunks, dim=2)
        print(f">>> Full-res latent shape: {full_res_latent.shape}")

        # Calculate how many latent frames to preserve (same formula as create_av_noise_mask)
        video_start_idx = time_to_video_latent_idx(
            start_time, output_fps, time_scale_factor, full_res_latent.shape[2]
        )

        # Replace preserved frames AND transition zone in upscaled latent with original full-res encoding
        # Include slope_len extra frames to cover the gradient transition zone where mask < 1.0
        # This ensures clean_latent is correct for partial preservation blending
        preserve_end_idx = min(video_start_idx + slope_len, full_res_latent.shape[2])
        print(f">>> Replacing frames 0-{preserve_end_idx} (including transition zone) with original full-res encoding...")
        upscaled_video_latent[:, :, :preserve_end_idx, :, :] = full_res_latent[:, :, :preserve_end_idx, :, :]

        # Cleanup
        del full_res_frames, full_res_tensor, full_res_input, full_res_latent_chunks, full_res_latent
        cleanup_memory()

        # 9B: Load stage 2 transformer with distilled LoRA
        print(">>> Loading stage 2 transformer...")
        stage2_block_swap_manager = None
        if generator.enable_refiner_block_swap:
            print(f">>> Loading stage 2 transformer with block swapping ({generator.refiner_blocks_in_memory} blocks in GPU)...", flush=True)
            from ltx_core.loader.sft_loader import SafetensorsStateDictLoader

            # Create ledger without LoRAs for fast base model loading
            stage_2_ledger_no_lora = ModelLedger(
                dtype=generator.dtype,
                device=torch.device("cpu"),
                checkpoint_path=generator.stage_2_model_ledger.checkpoint_path if hasattr(generator.stage_2_model_ledger, 'checkpoint_path') else generator.stage_1_model_ledger.checkpoint_path,
                gemma_root_path=generator.stage_2_model_ledger.gemma_root_path if hasattr(generator.stage_2_model_ledger, 'gemma_root_path') else generator.stage_1_model_ledger.gemma_root_path,
                spatial_upsampler_path=generator.stage_2_model_ledger.spatial_upsampler_path if hasattr(generator.stage_2_model_ledger, 'spatial_upsampler_path') else generator.stage_1_model_ledger.spatial_upsampler_path,
                vae_path=generator.stage_2_model_ledger.vae_path if hasattr(generator.stage_2_model_ledger, 'vae_path') else generator.stage_1_model_ledger.vae_path,
                loras=(),  # No LoRAs - load base model only
                fp8transformer=generator.stage_2_model_ledger.fp8transformer if hasattr(generator.stage_2_model_ledger, 'fp8transformer') else generator.stage_1_model_ledger.fp8transformer,
            )
            stage2_transformer = stage_2_ledger_no_lora.transformer()

            # Apply LoRAs (distilled + user LoRAs) using chunked GPU computation
            if hasattr(generator.stage_2_model_ledger, 'loras') and generator.stage_2_model_ledger.loras:
                loras = generator.stage_2_model_ledger.loras
                print(f">>> Applying {len(loras)} LoRA(s) for stage 2...", flush=True)
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

            # Move non-block components to GPU
            stage2_transformer.velocity_model.patchify_proj.to(device)
            stage2_transformer.velocity_model.adaln_single.to(device)
            stage2_transformer.velocity_model.caption_projection.to(device)
            stage2_transformer.velocity_model.norm_out.to(device)
            stage2_transformer.velocity_model.proj_out.to(device)
            stage2_transformer.velocity_model.scale_shift_table = torch.nn.Parameter(
                stage2_transformer.velocity_model.scale_shift_table.to(device)
            )
            if hasattr(stage2_transformer.velocity_model, "audio_patchify_proj"):
                stage2_transformer.velocity_model.audio_patchify_proj.to(device)
            if hasattr(stage2_transformer.velocity_model, "audio_adaln_single"):
                stage2_transformer.velocity_model.audio_adaln_single.to(device)
            if hasattr(stage2_transformer.velocity_model, "audio_caption_projection"):
                stage2_transformer.velocity_model.audio_caption_projection.to(device)
            if hasattr(stage2_transformer.velocity_model, "audio_norm_out"):
                stage2_transformer.velocity_model.audio_norm_out.to(device)
            if hasattr(stage2_transformer.velocity_model, "audio_proj_out"):
                stage2_transformer.velocity_model.audio_proj_out.to(device)
            if hasattr(stage2_transformer.velocity_model, "audio_scale_shift_table"):
                stage2_transformer.velocity_model.audio_scale_shift_table = torch.nn.Parameter(
                    stage2_transformer.velocity_model.audio_scale_shift_table.to(device)
                )
            if hasattr(stage2_transformer.velocity_model, "av_ca_video_scale_shift_adaln_single"):
                stage2_transformer.velocity_model.av_ca_video_scale_shift_adaln_single.to(device)
            if hasattr(stage2_transformer.velocity_model, "av_ca_audio_scale_shift_adaln_single"):
                stage2_transformer.velocity_model.av_ca_audio_scale_shift_adaln_single.to(device)
            if hasattr(stage2_transformer.velocity_model, "av_ca_a2v_gate_adaln_single"):
                stage2_transformer.velocity_model.av_ca_a2v_gate_adaln_single.to(device)
            if hasattr(stage2_transformer.velocity_model, "av_ca_v2a_gate_adaln_single"):
                stage2_transformer.velocity_model.av_ca_v2a_gate_adaln_single.to(device)

            # Use activation offload for extreme memory savings
            if getattr(generator, 'enable_activation_offload', False):
                stage2_block_swap_manager = enable_block_swap_with_activation_offload(
                    stage2_transformer,
                    blocks_in_memory=generator.refiner_blocks_in_memory,
                    device=device,
                    verbose=True,
                    temporal_chunk_size=getattr(generator, 'temporal_chunk_size', 0),
                )
            else:
                stage2_block_swap_manager = enable_block_swap(
                    stage2_transformer,
                    blocks_in_memory=generator.refiner_blocks_in_memory,
                    device=device,
                )
        else:
            stage2_transformer = generator.stage_2_model_ledger.transformer()

        # 9C: Stage 2 sigma schedule (pre-tuned for distilled model)
        STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]
        num_stage2_sigmas = min(args.stage2_steps + 1, len(STAGE_2_DISTILLED_SIGMA_VALUES))
        stage2_sigmas = torch.tensor(
            STAGE_2_DISTILLED_SIGMA_VALUES[:num_stage2_sigmas],
            dtype=torch.float32, device=device
        )

        # 9D: Create stage 2 output shape (2x upscaled from stage 1)
        # Note: Use actual upscaled dimensions, not args.height/width
        # because stage1 dimensions may be rounded to be divisible by 32
        stage2_output_shape = VideoPixelShape(
            batch=1,
            frames=output_frames,
            height=stage1_height * 2,  # 2x stage 1 height
            width=stage1_width * 2,    # 2x stage 1 width
            fps=output_fps,
        )

        # 9E: Use the generator's pipeline components
        stage2_components = generator.pipeline_components

        # 9F: Stage 2 denoising function (NO CFG, just positive context)
        stage2_denoise_fn = simple_denoising_func(
            video_context=v_context_p,
            audio_context=a_context_p,
            transformer=stage2_transformer,
        )

        # 9G: Define stage 2 denoising loop
        def stage2_denoising_loop(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: DiffusionStepProtocol,
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=stage2_denoise_fn,
                latent_norm_fn=latent_norm_fn,
            )

        # 9H: Run stage 2 denoising WITH PRESERVATION MASK
        # Unlike denoise_audio_video which creates all-ones mask, we manually apply
        # the preservation mask to prevent re-denoising the input video frames
        print(f">>> Stage 2 denoising with {len(stage2_sigmas) - 1} steps (with preservation mask)...")

        # Upscale video_mask to stage 2 resolution (2x spatial)
        # video_mask is [B, 1, F, H, W] from stage 1
        stage2_video_mask = torch.nn.functional.interpolate(
            video_mask.to(device=device, dtype=torch.float32),
            scale_factor=(1, 2, 2),  # Keep F, 2x H and W
            mode='nearest'  # Binary mask, use nearest neighbor
        )

        # Create stage 2 video latent tools
        stage2_video_latent_shape = VideoLatentShape.from_pixel_shape(
            shape=stage2_output_shape,
            latent_channels=stage2_components.video_latent_channels,
            scale_factors=stage2_components.video_scale_factors,
        )
        stage2_video_tools = VideoLatentTools(
            patchifier=stage2_components.video_patchifier,
            target_shape=stage2_video_latent_shape,
            fps=output_fps,
        )

        # Create initial video state with upscaled latent
        stage2_video_state = stage2_video_tools.create_initial_state(
            device, dtype, upscaled_video_latent
        )

        # Apply preservation mask (patchify and replace)
        patchified_stage2_video_mask = stage2_video_tools.patchifier.patchify(stage2_video_mask)
        stage2_video_state = dataclass_replace(
            stage2_video_state,
            denoise_mask=patchified_stage2_video_mask.to(dtype=torch.float32),
        )

        # Apply noiser WITH mask - preserved frames won't get noise
        stage2_video_state = noiser(stage2_video_state, noise_scale=stage2_sigmas[0].item())

        if denoised_audio_latent is not None:
            stage2_audio_latent_shape = AudioLatentShape.from_torch_shape(denoised_audio_latent.shape)
        else:
            stage2_audio_latent_shape = AudioLatentShape.from_video_pixel_shape(stage2_output_shape)

        stage2_audio_tools = AudioLatentTools(
            patchifier=stage2_components.audio_patchifier,
            target_shape=stage2_audio_latent_shape,
        )

        stage2_audio_state = stage2_audio_tools.create_initial_state(
            device, dtype, denoised_audio_latent
        )
        stage2_audio_state = dataclass_replace(
            stage2_audio_state,
            denoise_mask=torch.zeros_like(stage2_audio_state.denoise_mask),
        )
        stage2_audio_state = noiser(stage2_audio_state, noise_scale=stage2_sigmas[0].item())

        # Run stage 2 denoising loop with masked states
        final_stage2_video, final_stage2_audio = euler_denoising_loop(
            sigmas=stage2_sigmas,
            video_state=stage2_video_state,
            audio_state=stage2_audio_state,
            stepper=stepper,
            denoise_fn=stage2_denoise_fn,
            latent_norm_fn=latent_norm_fn,
        )

        # Unpatchify results
        final_stage2_video = stage2_video_tools.clear_conditioning(final_stage2_video)
        final_stage2_video = stage2_video_tools.unpatchify(final_stage2_video)
        final_stage2_audio = stage2_audio_tools.clear_conditioning(final_stage2_audio)
        final_stage2_audio = stage2_audio_tools.unpatchify(final_stage2_audio)

        # 9I: Get stage 2 results
        denoised_video_latent = final_stage2_video.latent
        if denoised_audio_latent is not None:
            denoised_audio_latent = final_stage2_audio.latent

        # Cleanup stage 2 transformer with proper block swap handling
        if stage2_block_swap_manager is not None:
            offload_all_blocks(stage2_transformer)
            stage2_transformer.velocity_model._block_swap_offloader = None
            stage2_transformer.velocity_model._blocks_ref = None
            stage2_block_swap_manager = None
        del stage2_transformer, video_encoder
        cleanup_memory()

    # =========================================================================
    # Step 10: Decode final video (after stage 2 if enabled)
    # =========================================================================
    print(">>> Decoding video...")

    # Use stage 2 decoder for full resolution if stage 2 was run
    if not skip_stage2 and not generator.one_stage:
        video_decoder = generator.stage_2_model_ledger.video_decoder()
    else:
        video_decoder = generator.stage_1_model_ledger.video_decoder()

    tiling_config = TilingConfig.default()
    # Collect decoded chunks on CPU immediately to save GPU memory during tiled decoding
    decoded_video_chunks = []
    for chunk in vae_decode_video(
        denoised_video_latent,  # Pass directly, decoder handles dtype
        video_decoder,
        tiling_config,
    ):
        decoded_video_chunks.append(chunk.cpu())  # Move to CPU immediately
        del chunk
    decoded_video = torch.cat(decoded_video_chunks, dim=0)  # [F, H, W, C] on CPU

    del video_decoder, decoded_video_chunks
    cleanup_memory()

    # Decode audio if present
    decoded_audio = None
    if denoised_audio_latent is not None:
        print(">>> Decoding audio...")
        audio_decoder = generator.stage_1_model_ledger.audio_decoder()
        vocoder = generator.stage_1_model_ledger.vocoder()
        decoded_audio = vae_decode_audio(
            denoised_audio_latent,  # Pass directly, decoder handles dtype
            audio_decoder,
            vocoder,
        )
        del audio_decoder, vocoder
        cleanup_memory()

    print(f">>> Output video shape: {decoded_video.shape}")
    if decoded_audio is not None:
        print(f">>> Output audio shape: {decoded_audio.shape}")

    return decoded_video, decoded_audio


# =============================================================================
# V2V Join Generation (Video-to-Video Transition)
# =============================================================================

def generate_v2v_join(
    generator: "LTXVideoGeneratorWithOffloading",
    args,
    video1_path: str,
    video2_path: str,
    frames_check1: int = 30,
    frames_check2: int = 30,
    preserve1_sec: float = 5.0,
    preserve2_sec: float = 5.0,
    transition_sec: float = 10.0,
    extend_steps: int = 8,
    terminal: float = 0.1,
    slope_len: int = 3,
    skip_stage2: bool = False,
    latent_norm_fn: Callable | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Join two videos by generating a smooth transition between them.

    This function:
    1. Finds optimal transition points in both videos using sharpness detection
    2. Preserves sections from the end of video1 and start of video2
    3. Generates a transition between them using masked denoising
    4. Concatenates everything into a final joined video

    Args:
        generator: LTXVideoGeneratorWithOffloading instance
        args: Command line arguments
        video1_path: Path to first video
        video2_path: Path to second video
        frames_check1: Number of frames to check for sharpness at end of video1
        frames_check2: Number of frames to check for sharpness at start of video2
        preserve1_sec: Seconds to preserve from end of video1
        preserve2_sec: Seconds to preserve from start of video2
        transition_sec: Total seconds for generated transition between preserved sections
        extend_steps: Number of denoising steps
        terminal: Terminal sigma for partial denoising
        slope_len: Transition slope length at mask boundaries
        skip_stage2: Whether to skip stage 2 refinement
        latent_norm_fn: Optional latent normalization function

    Returns:
        Tuple of (video_tensor [F, H, W, C], audio_tensor or None)
    """
    import cv2
    import numpy as np
    from dataclasses import replace as dataclass_replace
    from ltx_core.types import LatentState, VideoPixelShape

    device = generator.device
    dtype = generator.dtype

    print("=" * 60)
    print("V2V Join Mode (Video-to-Video Transition)")
    print("=" * 60)

    # =========================================================================
    # Step 1: Analyze both videos
    # =========================================================================
    print(">>> Analyzing input videos...")

    cap1 = cv2.VideoCapture(video1_path)
    if not cap1.isOpened():
        raise RuntimeError(f"Failed to open video1: {video1_path}")
    total_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration1 = total_frames1 / fps1
    cap1.release()

    cap2 = cv2.VideoCapture(video2_path)
    if not cap2.isOpened():
        raise RuntimeError(f"Failed to open video2: {video2_path}")
    total_frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration2 = total_frames2 / fps2
    cap2.release()

    print(f">>> Video1: {total_frames1} frames at {fps1:.1f} fps, {width1}x{height1}, {duration1:.2f}s")
    print(f">>> Video2: {total_frames2} frames at {fps2:.1f} fps, {width2}x{height2}, {duration2:.2f}s")

    # =========================================================================
    # Step 2: Find best transition frames using sharpness detection
    # =========================================================================
    best_frame1 = extract_best_transition_frame_from_end(video1_path, frames_check1)
    best_frame2 = extract_best_transition_frame_from_start(video2_path, frames_check2)

    if best_frame1 < 0:
        best_frame1 = total_frames1 - 1
        print(f">>> Using last frame of video1: {best_frame1}")

    # =========================================================================
    # Step 3: Calculate segment boundaries
    # =========================================================================
    output_fps = args.frame_rate

    # Frames to preserve from each video (in output fps)
    preserve1_frames = int(round(preserve1_sec * output_fps))
    preserve2_frames = int(round(preserve2_sec * output_fps))

    # Ensure 8n+1 format for VAE
    preserve1_frames = ((preserve1_frames - 1) // 8) * 8 + 1
    preserve2_frames = ((preserve2_frames - 1) // 8) * 8 + 1

    # Total transition frames (preserved1 + generated + preserved2)
    total_transition_frames = int(round((preserve1_sec + transition_sec + preserve2_sec) * output_fps))
    total_transition_frames = ((total_transition_frames - 1) // 8) * 8 + 1

    # Generated frames = total - preserved sections
    generated_frames = total_transition_frames - preserve1_frames - preserve2_frames
    if generated_frames < 9:
        # Minimum 9 frames for generation
        generated_frames = 9
        total_transition_frames = preserve1_frames + generated_frames + preserve2_frames
        total_transition_frames = ((total_transition_frames - 1) // 8) * 8 + 1

    print(f">>> Transition structure:")
    print(f">>>   Preserve from video1: {preserve1_frames} frames ({preserve1_frames / output_fps:.2f}s)")
    print(f">>>   Generate transition: {generated_frames} frames ({generated_frames / output_fps:.2f}s)")
    print(f">>>   Preserve from video2: {preserve2_frames} frames ({preserve2_frames / output_fps:.2f}s)")
    print(f">>>   Total transition: {total_transition_frames} frames ({total_transition_frames / output_fps:.2f}s)")

    # Calculate which frames to extract from source videos
    # Video1: extract preserve1_frames ending at best_frame1
    v1_start_frame = max(0, best_frame1 - int(preserve1_frames * fps1 / output_fps) + 1)
    v1_end_frame = best_frame1 + 1  # +1 because end is exclusive

    # Video2: extract preserve2_frames starting at best_frame2
    v2_start_frame = best_frame2
    v2_end_frame = min(total_frames2, best_frame2 + int(preserve2_frames * fps2 / output_fps))

    # Also calculate prefix (before transition) and suffix (after transition)
    v1_prefix_end = v1_start_frame  # All frames before transition start
    v2_suffix_start = v2_end_frame  # All frames after transition end

    print(f">>> Video1 extraction: frames {v1_start_frame}-{v1_end_frame} (prefix ends at {v1_prefix_end})")
    print(f">>> Video2 extraction: frames {v2_start_frame}-{v2_end_frame} (suffix starts at {v2_suffix_start})")

    # =========================================================================
    # Step 4: Load video segments
    # =========================================================================
    print(">>> Loading video segments...")

    # Determine output resolution
    out_width = args.width
    out_height = args.height

    # Use half resolution for two-stage pipeline
    if not generator.one_stage:
        stage1_width = out_width // 2
        stage1_height = out_height // 2
    else:
        stage1_width = out_width
        stage1_height = out_height

    # Ensure dimensions are divisible by 32
    stage1_width = (stage1_width // 32) * 32
    stage1_height = (stage1_height // 32) * 32

    # Load preserved section from video1 (end section)
    frames1, _ = load_video_segment(
        video1_path, v1_start_frame, v1_end_frame,
        stage1_width, stage1_height, device, dtype
    )
    print(f">>> Loaded video1 segment: {frames1.shape}")

    # Load preserved section from video2 (start section)
    frames2, _ = load_video_segment(
        video2_path, v2_start_frame, v2_end_frame,
        stage1_width, stage1_height, device, dtype
    )
    print(f">>> Loaded video2 segment: {frames2.shape}")

    # Resample to match output fps and target frame counts
    # Video1 preserved frames
    if frames1.shape[0] != preserve1_frames:
        indices = torch.linspace(0, frames1.shape[0] - 1, preserve1_frames).long()
        frames1 = frames1[indices]
        print(f">>> Resampled video1 segment to {frames1.shape[0]} frames")

    # Video2 preserved frames
    if frames2.shape[0] != preserve2_frames:
        indices = torch.linspace(0, frames2.shape[0] - 1, preserve2_frames).long()
        frames2 = frames2[indices]
        print(f">>> Resampled video2 segment to {frames2.shape[0]} frames")

    # =========================================================================
    # Step 5: Create combined input for transition
    # =========================================================================
    # The transition region will be: [preserved1 | zeros_for_generation | preserved2]
    # We need to create a tensor with the full transition length

    # Create empty frames for generation region (initialized with interpolation)
    print(">>> Creating transition frame buffer...")

    # Linear interpolation between last frame of video1 and first frame of video2
    # for the generation region as initialization
    last_frame1 = frames1[-1:].clone()
    first_frame2 = frames2[:1].clone()

    # Create interpolated frames for generation region
    gen_frames = []
    for i in range(generated_frames):
        alpha = i / max(generated_frames - 1, 1)
        interp_frame = last_frame1 * (1 - alpha) + first_frame2 * alpha
        gen_frames.append(interp_frame)
    gen_frames_tensor = torch.cat(gen_frames, dim=0)

    # Combine all frames: [preserved1 | interpolated_gen | preserved2]
    combined_frames = torch.cat([frames1, gen_frames_tensor, frames2], dim=0)
    print(f">>> Combined frame buffer: {combined_frames.shape}")

    # Free memory
    del frames1, frames2, gen_frames_tensor, gen_frames
    cleanup_memory()

    # =========================================================================
    # Step 6: Encode to latent space
    # =========================================================================
    print(">>> Encoding frames to latent space...")

    video_encoder = generator.stage_1_model_ledger.video_encoder()
    encoder_dtype = next(video_encoder.parameters()).dtype
    time_scale_factor = video_encoder.downscale_index_formula[0] if hasattr(video_encoder, 'downscale_index_formula') else 8

    # Convert from [F, H, W, C] to [1, C, F, H, W] and normalize to [-1, 1]
    video_input = combined_frames.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, F, H, W]
    video_input = video_input * 2.0 - 1.0

    del combined_frames
    cleanup_memory()

    # Encode in temporal chunks to avoid OOM
    chunk_pixel_frames = 65  # 8*8+1 = 65 frames per chunk
    total_pixel_frames = video_input.shape[2]

    latent_chunks = []
    chunk_idx = 0

    with torch.no_grad():
        for start_frame in range(0, total_pixel_frames, chunk_pixel_frames - 1):
            end_frame = min(start_frame + chunk_pixel_frames, total_pixel_frames)
            actual_frames = end_frame - start_frame

            # Ensure we have at least 9 frames (minimum for VAE)
            if actual_frames < 9:
                pad_frames = 9 - actual_frames
                chunk = video_input[:, :, start_frame:end_frame, :, :]
                last_frame = chunk[:, :, -1:, :, :].expand(-1, -1, pad_frames, -1, -1)
                chunk = torch.cat([chunk, last_frame], dim=2)
            else:
                chunk = video_input[:, :, start_frame:end_frame, :, :]

            chunk_latent = video_encoder(chunk.to(device=device, dtype=encoder_dtype))
            chunk_latent = chunk_latent.to(dtype=dtype)

            # Skip first latent frame for non-first chunks (temporal overlap)
            if chunk_idx > 0 and len(latent_chunks) > 0:
                chunk_latent = chunk_latent[:, :, 1:, :, :]

            latent_chunks.append(chunk_latent)
            chunk_idx += 1

            print(f">>> Encoded chunk {chunk_idx}: frames {start_frame}-{end_frame} -> {chunk_latent.shape[2]} latent frames")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    video_latent = torch.cat(latent_chunks, dim=2)
    print(f">>> Full video latent: {video_latent.shape}")

    # Free encoder
    video_encoder.to("cpu")
    del video_encoder, latent_chunks, video_input
    cleanup_memory()

    # =========================================================================
    # Step 7: Create noise mask for V2V join
    # =========================================================================
    print(">>> Creating V2V join noise mask...")

    B, C, F_latent, H_latent, W_latent = video_latent.shape

    # Calculate latent frame indices for preserved/generation boundaries
    # preserve1_frames -> latent frames
    preserve1_latent = (preserve1_frames - 1) // time_scale_factor + 1
    # preserve2_frames -> latent frames
    preserve2_latent = (preserve2_frames - 1) // time_scale_factor + 1
    # Generation region is everything in between
    gen_start_latent = preserve1_latent
    gen_end_latent = F_latent - preserve2_latent

    print(f">>> Latent mask: preserve1=0-{gen_start_latent}, gen={gen_start_latent}-{gen_end_latent}, preserve2={gen_end_latent}-{F_latent}")

    # Create mask: 0 = preserve (no denoising), 1 = generate (full denoising)
    video_mask = torch.zeros((B, 1, F_latent, H_latent, W_latent), device=device, dtype=torch.float32)

    # Set generation region to 1
    video_mask[:, :, gen_start_latent:gen_end_latent, :, :] = 1.0

    # Apply slope at boundaries for smooth blending
    if slope_len > 0:
        # Slope at start of generation (transition from preserve1 to gen)
        for i in range(min(slope_len, gen_end_latent - gen_start_latent)):
            slope_value = (i + 1) / (slope_len + 1)
            video_mask[:, :, gen_start_latent + i, :, :] = slope_value

        # Slope at end of generation (transition from gen to preserve2)
        for i in range(min(slope_len, gen_end_latent - gen_start_latent)):
            slope_value = (i + 1) / (slope_len + 1)
            video_mask[:, :, gen_end_latent - 1 - i, :, :] = slope_value

    print(f">>> Video mask created with slope_len={slope_len}")

    # =========================================================================
    # Step 8: Run masked denoising (similar to AV extension)
    # =========================================================================
    print(">>> Running masked denoising...")

    # Offload latents to CPU for model loading
    video_latent = video_latent.cpu()
    video_mask = video_mask.cpu()
    cleanup_memory()

    # Load text encoder and encode prompts
    text_encoder = generator.stage_1_model_ledger.text_encoder()
    print(">>> Encoding prompts...")
    if args.cfg_guidance_scale > 1.0:
        context_p, context_n = encode_text(text_encoder, prompts=[args.prompt, args.negative_prompt])
        v_context_p, a_context_p = context_p
        v_context_n, a_context_n = context_n
    else:
        context_p = encode_text(text_encoder, prompts=[args.prompt])[0]
        v_context_p, a_context_p = context_p
        v_context_n, a_context_n = None, None

    del text_encoder
    cleanup_memory()

    # Reload latents to GPU
    video_latent = video_latent.to(device=device, dtype=dtype)
    video_mask = video_mask.to(device=device)

    # Load transformer
    block_swap_manager = None
    if generator.enable_dit_block_swap:
        print(f">>> Loading DiT transformer with block swapping ({generator.dit_blocks_in_memory} blocks in GPU)...")
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

        if hasattr(transformer.velocity_model, "audio_patchify_proj"):
            transformer.velocity_model.audio_patchify_proj.to(device)
        if hasattr(transformer.velocity_model, "audio_adaln_single"):
            transformer.velocity_model.audio_adaln_single.to(device)
        if hasattr(transformer.velocity_model, "audio_caption_projection"):
            transformer.velocity_model.audio_caption_projection.to(device)
        if hasattr(transformer.velocity_model, "audio_norm_out"):
            transformer.velocity_model.audio_norm_out.to(device)
        if hasattr(transformer.velocity_model, "audio_proj_out"):
            transformer.velocity_model.audio_proj_out.to(device)
        if hasattr(transformer.velocity_model, "audio_scale_shift_table"):
            transformer.velocity_model.audio_scale_shift_table = torch.nn.Parameter(
                transformer.velocity_model.audio_scale_shift_table.to(device)
            )

        if hasattr(transformer.velocity_model, "av_ca_video_scale_shift_adaln_single"):
            transformer.velocity_model.av_ca_video_scale_shift_adaln_single.to(device)
        if hasattr(transformer.velocity_model, "av_ca_audio_scale_shift_adaln_single"):
            transformer.velocity_model.av_ca_audio_scale_shift_adaln_single.to(device)
        if hasattr(transformer.velocity_model, "av_ca_a2v_gate_adaln_single"):
            transformer.velocity_model.av_ca_a2v_gate_adaln_single.to(device)
        if hasattr(transformer.velocity_model, "av_ca_v2a_gate_adaln_single"):
            transformer.velocity_model.av_ca_v2a_gate_adaln_single.to(device)

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

    # Create sigmas schedule
    sigmas = LTX2Scheduler().execute(
        steps=extend_steps,
        latent=video_latent,
        terminal=terminal,
        stretch=True,
    ).to(dtype=torch.float32, device=device)

    # Initialize diffusion components
    generator_torch = torch.Generator(device=device).manual_seed(args.seed)
    noiser = GaussianNoiser(generator=generator_torch)
    stepper = EulerDiffusionStep()
    cfg_guider = CFGGuider(args.cfg_guidance_scale)
    effective_stg_blocks = args.stg_blocks if args.stg_blocks is not None else [29]
    stg_guider = STGGuider(args.stg_scale)
    stg_perturbation_config = build_stg_perturbation_config(
        stg_scale=args.stg_scale,
        stg_blocks=effective_stg_blocks,
        stg_mode=args.stg_mode,
    )

    # Get components
    components = generator.pipeline_components

    # Create the denoise function (CFG + STG combined)
    use_cfg = args.cfg_guidance_scale > 1.0 and v_context_n is not None
    use_stg = stg_guider.enabled() and stg_perturbation_config is not None
    if use_cfg or use_stg:
        denoise_fn = cfg_stg_denoising_func(
            cfg_guider=cfg_guider,
            stg_guider=stg_guider,
            stg_perturbation_config=stg_perturbation_config,
            v_context_p=v_context_p,
            v_context_n=v_context_n,
            a_context_p=a_context_p,
            a_context_n=a_context_n,
            transformer=transformer,
        )
    else:
        denoise_fn = simple_denoising_func(v_context_p, a_context_p, transformer)

    from ltx_core.types import VideoLatentShape, AudioLatentShape
    from ltx_core.tools import VideoLatentTools, AudioLatentTools
    from ltx_pipelines.utils.helpers import euler_denoising_loop

    # Create video shape and latent tools
    output_shape = VideoPixelShape(
        batch=1,
        frames=total_transition_frames,
        height=stage1_height,
        width=stage1_width,
        fps=output_fps,
    )
    video_latent_shape = VideoLatentShape.from_pixel_shape(
        shape=output_shape,
        latent_channels=components.video_latent_channels,
        scale_factors=components.video_scale_factors,
    )
    video_tools = VideoLatentTools(
        patchifier=components.video_patchifier,
        target_shape=video_latent_shape,
        fps=output_fps,
    )

    # Create initial video state with mask
    video_state = video_tools.create_initial_state(device, dtype, video_latent)

    # Patchify the mask
    patchified_video_mask = video_tools.patchifier.patchify(video_mask)
    video_state = dataclass_replace(
        video_state,
        denoise_mask=patchified_video_mask.to(dtype=torch.float32),
    )

    # Add noise based on mask
    video_state = noiser(video_state, noise_scale=1.0)

    # Create dummy audio state (no audio generation for V2V join)
    audio_latent_shape = AudioLatentShape.from_video_pixel_shape(shape=output_shape)
    audio_tools = AudioLatentTools(
        patchifier=components.audio_patchifier,
        target_shape=audio_latent_shape,
    )
    audio_state = audio_tools.create_initial_state(device, dtype, None)
    audio_state = dataclass_replace(
        audio_state,
        denoise_mask=torch.zeros_like(audio_state.denoise_mask),
    )
    audio_state = noiser(audio_state, noise_scale=1.0)

    # Run denoising loop
    final_video_state, final_audio_state = euler_denoising_loop(
        sigmas=sigmas,
        video_state=video_state,
        audio_state=audio_state,
        stepper=stepper,
        denoise_fn=denoise_fn,
        latent_norm_fn=latent_norm_fn,
    )

    # Unpatchify
    final_video_state = video_tools.clear_conditioning(final_video_state)
    final_video_state = video_tools.unpatchify(final_video_state)

    denoised_video_latent = final_video_state.latent
    print(f">>> Denoised video latent: {denoised_video_latent.shape}")

    # Cleanup transformer
    if block_swap_manager is not None:
        offload_all_blocks(transformer)
        transformer.velocity_model._block_swap_offloader = None
        transformer.velocity_model._blocks_ref = None
        block_swap_manager = None
    del transformer
    cleanup_memory()
    phase_barrier("v2v_join_denoising")

    # =========================================================================
    # Step 9: Stage 2 Refinement (if enabled)
    # =========================================================================
    if not skip_stage2 and not generator.one_stage:
        print(">>> Stage 2 refinement...")

        from ltx_core.model.upsampler import upsample_video as upsample_latent_fn

        video_encoder = generator.stage_1_model_ledger.video_encoder()
        spatial_upsampler = generator.stage_2_model_ledger.spatial_upsampler()

        print(">>> Upsampling latents (2x)...")
        upscaled_video_latent = upsample_latent_fn(
            latent=denoised_video_latent,
            video_encoder=video_encoder,
            upsampler=spatial_upsampler,
        )

        del spatial_upsampler
        cleanup_memory()

        # Stage 2 transformer with refinement
        print(">>> Loading stage 2 transformer...")
        stage2_block_swap_manager = None

        if generator.enable_refiner_block_swap:
            from ltx_core.loader.sft_loader import SafetensorsStateDictLoader

            stage_2_ledger_no_lora = ModelLedger(
                dtype=generator.dtype,
                device=torch.device("cpu"),
                checkpoint_path=generator.stage_2_model_ledger.checkpoint_path if hasattr(generator.stage_2_model_ledger, 'checkpoint_path') else generator.stage_1_model_ledger.checkpoint_path,
                gemma_root_path=generator.stage_2_model_ledger.gemma_root_path if hasattr(generator.stage_2_model_ledger, 'gemma_root_path') else generator.stage_1_model_ledger.gemma_root_path,
                spatial_upsampler_path=generator.stage_2_model_ledger.spatial_upsampler_path if hasattr(generator.stage_2_model_ledger, 'spatial_upsampler_path') else generator.stage_1_model_ledger.spatial_upsampler_path,
                vae_path=generator.stage_2_model_ledger.vae_path if hasattr(generator.stage_2_model_ledger, 'vae_path') else generator.stage_1_model_ledger.vae_path,
                loras=(),
                fp8transformer=generator.stage_2_model_ledger.fp8transformer if hasattr(generator.stage_2_model_ledger, 'fp8transformer') else generator.stage_1_model_ledger.fp8transformer,
            )
            stage2_transformer = stage_2_ledger_no_lora.transformer()

            if hasattr(generator.stage_2_model_ledger, 'loras') and generator.stage_2_model_ledger.loras:
                loras = generator.stage_2_model_ledger.loras
                print(f">>> Applying {len(loras)} LoRA(s) for stage 2...")
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

            # Move non-block components to GPU
            stage2_transformer.velocity_model.patchify_proj.to(device)
            stage2_transformer.velocity_model.adaln_single.to(device)
            stage2_transformer.velocity_model.caption_projection.to(device)
            stage2_transformer.velocity_model.norm_out.to(device)
            stage2_transformer.velocity_model.proj_out.to(device)
            stage2_transformer.velocity_model.scale_shift_table = torch.nn.Parameter(
                stage2_transformer.velocity_model.scale_shift_table.to(device)
            )
            if hasattr(stage2_transformer.velocity_model, "audio_patchify_proj"):
                stage2_transformer.velocity_model.audio_patchify_proj.to(device)
            if hasattr(stage2_transformer.velocity_model, "audio_adaln_single"):
                stage2_transformer.velocity_model.audio_adaln_single.to(device)
            if hasattr(stage2_transformer.velocity_model, "audio_caption_projection"):
                stage2_transformer.velocity_model.audio_caption_projection.to(device)
            if hasattr(stage2_transformer.velocity_model, "audio_norm_out"):
                stage2_transformer.velocity_model.audio_norm_out.to(device)
            if hasattr(stage2_transformer.velocity_model, "audio_proj_out"):
                stage2_transformer.velocity_model.audio_proj_out.to(device)
            if hasattr(stage2_transformer.velocity_model, "audio_scale_shift_table"):
                stage2_transformer.velocity_model.audio_scale_shift_table = torch.nn.Parameter(
                    stage2_transformer.velocity_model.audio_scale_shift_table.to(device)
                )
            if hasattr(stage2_transformer.velocity_model, "av_ca_video_scale_shift_adaln_single"):
                stage2_transformer.velocity_model.av_ca_video_scale_shift_adaln_single.to(device)
            if hasattr(stage2_transformer.velocity_model, "av_ca_audio_scale_shift_adaln_single"):
                stage2_transformer.velocity_model.av_ca_audio_scale_shift_adaln_single.to(device)
            if hasattr(stage2_transformer.velocity_model, "av_ca_a2v_gate_adaln_single"):
                stage2_transformer.velocity_model.av_ca_a2v_gate_adaln_single.to(device)
            if hasattr(stage2_transformer.velocity_model, "av_ca_v2a_gate_adaln_single"):
                stage2_transformer.velocity_model.av_ca_v2a_gate_adaln_single.to(device)

            # Use activation offload for extreme memory savings
            if getattr(generator, 'enable_activation_offload', False):
                stage2_block_swap_manager = enable_block_swap_with_activation_offload(
                    stage2_transformer,
                    blocks_in_memory=generator.refiner_blocks_in_memory,
                    device=device,
                    verbose=True,
                    temporal_chunk_size=getattr(generator, 'temporal_chunk_size', 0),
                )
            else:
                stage2_block_swap_manager = enable_block_swap(
                    stage2_transformer,
                    blocks_in_memory=generator.refiner_blocks_in_memory,
                    device=device,
                )
        else:
            stage2_transformer = generator.stage_2_model_ledger.transformer()

        # Stage 2 sigmas
        stage2_sigmas = LTX2Scheduler().execute(
            steps=args.stage2_steps,
            latent=upscaled_video_latent,
            terminal=0.0,
            stretch=True,
        ).to(dtype=torch.float32, device=device)

        # Stage 2 components
        stage2_components = generator.pipeline_components

        # Stage 2 output shape (2x spatial resolution)
        stage2_output_shape = VideoPixelShape(
            batch=1,
            frames=total_transition_frames,
            height=stage1_height * 2,
            width=stage1_width * 2,
            fps=output_fps,
        )

        stage2_video_latent_shape = VideoLatentShape.from_pixel_shape(
            shape=stage2_output_shape,
            latent_channels=stage2_components.video_latent_channels,
            scale_factors=stage2_components.video_scale_factors,
        )
        stage2_video_tools = VideoLatentTools(
            patchifier=stage2_components.video_patchifier,
            target_shape=stage2_video_latent_shape,
            fps=output_fps,
        )

        # Stage 2 denoising function (NO CFG, just positive context)
        stage2_denoise_fn = simple_denoising_func(
            video_context=v_context_p,
            audio_context=a_context_p,
            transformer=stage2_transformer,
        )

        # Create stage 2 video state
        stage2_video_state = stage2_video_tools.create_initial_state(device, dtype, upscaled_video_latent)
        stage2_noiser = GaussianNoiser(generator=torch.Generator(device=device).manual_seed(args.seed + 1))
        stage2_video_state = stage2_noiser(stage2_video_state, noise_scale=1.0)

        # Dummy audio state for stage 2
        stage2_audio_latent_shape = AudioLatentShape.from_video_pixel_shape(stage2_output_shape)
        stage2_audio_tools = AudioLatentTools(
            patchifier=stage2_components.audio_patchifier,
            target_shape=stage2_audio_latent_shape,
        )
        stage2_audio_state = stage2_audio_tools.create_initial_state(device, dtype, None)
        stage2_audio_state = stage2_noiser(stage2_audio_state, noise_scale=1.0)

        # Run stage 2 denoising
        final_stage2_video_state, _ = euler_denoising_loop(
            sigmas=stage2_sigmas,
            video_state=stage2_video_state,
            audio_state=stage2_audio_state,
            stepper=stepper,
            denoise_fn=stage2_denoise_fn,
            latent_norm_fn=latent_norm_fn,
        )

        final_stage2_video_state = stage2_video_tools.clear_conditioning(final_stage2_video_state)
        final_stage2_video_state = stage2_video_tools.unpatchify(final_stage2_video_state)

        denoised_video_latent = final_stage2_video_state.latent
        print(f">>> Stage 2 denoised latent: {denoised_video_latent.shape}")

        # Cleanup stage 2
        if stage2_block_swap_manager is not None:
            offload_all_blocks(stage2_transformer)
            stage2_transformer.velocity_model._block_swap_offloader = None
            stage2_transformer.velocity_model._blocks_ref = None
        del stage2_transformer
        cleanup_memory()

        del video_encoder
        cleanup_memory()
        phase_barrier("v2v_join_stage2")

    # =========================================================================
    # Step 10: Decode transition video
    # =========================================================================
    print(">>> Decoding transition video...")

    video_decoder = generator.stage_1_model_ledger.video_decoder() if generator.one_stage else generator.stage_2_model_ledger.video_decoder()
    tiling_config = TilingConfig.from_args(args)

    decoded_video_chunks = []
    for chunk in chunk_decode_video(
        denoised_video_latent,
        video_decoder,
        tiling_config,
    ):
        decoded_video_chunks.append(chunk.cpu())
        del chunk
    transition_video = torch.cat(decoded_video_chunks, dim=0)  # [F, H, W, C]

    del video_decoder, decoded_video_chunks, denoised_video_latent
    cleanup_memory()

    print(f">>> Transition video decoded: {transition_video.shape}")

    # =========================================================================
    # Step 11: Load prefix and suffix, concatenate final video
    # =========================================================================
    print(">>> Loading video prefix and suffix...")

    # Determine final output resolution
    final_width = args.width
    final_height = args.height

    # Load video1 prefix (all frames before preserved section)
    if v1_prefix_end > 0:
        v1_prefix, _ = load_video_segment(
            video1_path, 0, v1_prefix_end,
            final_width, final_height, torch.device("cpu"), torch.float32
        )
        # Convert from 0-1 to 0-255 uint8
        v1_prefix = (v1_prefix * 255).to(torch.uint8)
        print(f">>> Video1 prefix: {v1_prefix.shape}")
    else:
        v1_prefix = None

    # Load video2 suffix (all frames after preserved section)
    if v2_suffix_start < total_frames2:
        v2_suffix, _ = load_video_segment(
            video2_path, v2_suffix_start, total_frames2,
            final_width, final_height, torch.device("cpu"), torch.float32
        )
        # Convert from 0-1 to 0-255 uint8
        v2_suffix = (v2_suffix * 255).to(torch.uint8)
        print(f">>> Video2 suffix: {v2_suffix.shape}")
    else:
        v2_suffix = None

    # Resize transition video to final resolution if needed
    if transition_video.shape[1] != final_height or transition_video.shape[2] != final_width:
        print(f">>> Resizing transition from {transition_video.shape[2]}x{transition_video.shape[1]} to {final_width}x{final_height}")
        transition_resized = []
        for i in range(transition_video.shape[0]):
            frame = transition_video[i].numpy()
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame = (frame * 255).astype(np.uint8)
            frame_resized = cv2.resize(frame, (final_width, final_height), interpolation=cv2.INTER_LANCZOS4)
            transition_resized.append(torch.from_numpy(frame_resized))
        transition_video = torch.stack(transition_resized, dim=0)
    else:
        # Convert to uint8 if needed
        if transition_video.dtype != torch.uint8:
            transition_video = (transition_video * 255).to(torch.uint8)

    # Concatenate: prefix + transition + suffix
    parts = []
    if v1_prefix is not None:
        parts.append(v1_prefix)
    parts.append(transition_video.cpu())
    if v2_suffix is not None:
        parts.append(v2_suffix)

    final_video = torch.cat(parts, dim=0)
    print(f">>> Final joined video: {final_video.shape}")

    # Convert back to float for output compatibility
    final_video = final_video.float() / 255.0

    return final_video, None


# =============================================================================
# Preview Generation
# =============================================================================

def generate_preview(
    latents: torch.Tensor,
    max_height: int = 200,
    num_keyframes: int = 4,
    # Latent spatial dimensions for unpatchifying sequence format
    latent_frames: int = 0,
    latent_height: int = 0,
    latent_width: int = 0,
) -> "Image.Image | None":
    """
    Generate RGB preview from latent tensor without VAE decoding.

    Uses RGB factor convolution (same technique as Wan2GP) for fast,
    memory-efficient preview during denoising.

    Args:
        latents: Latent tensor - can be:
            - Spatial format: [B, C, T, H, W] or [C, T, H, W]
            - Sequence format: [B, seq_len, C] (patchified, needs dimensions to reshape)
        max_height: Maximum height of output preview
        num_keyframes: Number of temporal keyframes to show
        latent_frames: Number of latent frames (for unpatchifying sequence format)
        latent_height: Latent height (for unpatchifying sequence format)
        latent_width: Latent width (for unpatchifying sequence format)

    Returns:
        PIL Image showing preview grid, or None if failed
    """
    from PIL import Image
    import numpy as np

    try:
        # Handle various latent shapes
        ndim = latents.dim()

        # Detect sequence/patchified format (used internally by LTX-V transformer)
        # Shape is [batch, seq_len, hidden_dim] where hidden_dim is 128
        if ndim == 3:
            batch, seq_len, hidden = latents.shape
            if hidden == 128:
                # Sequence format [B, seq_len, 128]
                # Need spatial dimensions to unpatchify (Wan2GP approach)
                if latent_frames > 0 and latent_height > 0 and latent_width > 0:
                    expected_tokens = latent_frames * latent_height * latent_width
                    # Handle conditioning tokens (may be prepended/appended)
                    if seq_len >= expected_tokens:
                        # Take only the video tokens (first expected_tokens)
                        video_tokens = latents[:, :expected_tokens, :]
                        # Reshape: [B, T*H*W, 128] -> [B, 128, T, H, W]
                        video_tokens = video_tokens.transpose(1, 2)  # [B, 128, T*H*W]
                        latents = video_tokens.reshape(batch, 128, latent_frames, latent_height, latent_width)
                    else:
                        # Token count mismatch - skip preview
                        return None
                else:
                    # No spatial dimensions provided - cannot unpatchify
                    return None
            else:
                # Not 128 channels, treat as [C, H, W] -> [1, C, 1, H, W]
                latents = latents.unsqueeze(0).unsqueeze(2)
        elif ndim == 2:
            seq_len, hidden = latents.shape
            if hidden == 128:
                # Sequence format without batch - cannot preview without dimensions
                return None
            # Treat as [H, W] - unusual, skip
            return None
        elif ndim == 4:
            # Could be [B, C, H, W] or [C, T, H, W]
            # Assume [C, T, H, W] for video latents
            latents = latents.unsqueeze(0)
        elif ndim != 5:
            print(f"Warning: Unexpected latent dims: {ndim}, shape: {latents.shape}")
            return None

        B, C, T, H, W = latents.shape

        # Verify this is spatial format with 128 channels
        if C != 128:
            # Not in expected spatial format
            return None

        # Select keyframes (temporal subsampling)
        num_keyframes = min(T, num_keyframes)
        if num_keyframes <= 0:
            return None

        skip = max(1, T / num_keyframes)
        frame_indices = [min(int(i * skip), T - 1) for i in range(num_keyframes)]

        # Select frames
        selected = torch.stack([latents[:, :, i, :, :] for i in frame_indices], dim=2)

        # Create weight tensor from RGB factors [3, 128] -> [3, 128, 1, 1, 1]
        weight = torch.tensor(
            LATENT_RGB_FACTORS_LTXV,
            device=latents.device,
            dtype=latents.dtype,
        ).T.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        bias = torch.tensor(
            LATENT_RGB_FACTORS_BIAS_LTXV,
            device=latents.device,
            dtype=latents.dtype,
        )

        # Apply 3D convolution for latent-to-RGB: [B, 128, T, H, W] -> [B, 3, T, H, W]
        images = torch.nn.functional.conv3d(
            selected, weight, bias=bias, stride=1, padding=0
        )

        # Normalize to 0-255
        images = images.add_(1.0).mul_(127.5)
        images = images.detach().cpu()

        if images.dtype == torch.bfloat16:
            images = images.to(torch.float16)

        images = images.numpy().clip(0, 255).astype(np.uint8)

        # Rearrange to grid: [B, C, T, H, W] -> [B*H, T*W, C]
        # For batch=1: [1, 3, T, H, W] -> [H, T*W, 3]
        B, C, T_sel, H, W = images.shape
        images = images[0]  # Remove batch dim: [3, T, H, W]
        images = images.transpose(1, 2, 3, 0)  # [T, H, W, 3]
        images = images.transpose(0, 2, 1, 3)  # [T, W, H, 3]
        images = images.reshape(T_sel * W, H, 3).transpose(1, 0, 2)  # [H, T*W, 3]

        # Resize to max_height
        h, w, _ = images.shape
        scale = max_height / h
        pil_image = Image.fromarray(images)
        pil_image = pil_image.resize(
            (int(w * scale), int(h * scale)),
            resample=Image.Resampling.BILINEAR,
        )

        return pil_image

    except Exception as e:
        print(f"Warning: Preview generation failed: {e}")
        return None


def create_preview_callback(
    preview_dir: str,
    preview_interval: int = 1,
    max_height: int = 200,
    stage_name: str = "stage1",
    preview_suffix: str = "",
    # Latent spatial dimensions for unpatchifying sequence format
    latent_frames: int = 0,
    latent_height: int = 0,
    latent_width: int = 0,
):
    """
    Create a callback function for preview generation during denoising.

    Args:
        preview_dir: Directory to save preview images
        preview_interval: Save every N steps
        max_height: Maximum preview height
        stage_name: Prefix for filenames (e.g., "stage1", "stage2")
        preview_suffix: Unique suffix for MP4 preview filename
        latent_frames: Number of latent frames (for unpatchifying sequence format)
        latent_height: Latent height (for unpatchifying sequence format)
        latent_width: Latent width (for unpatchifying sequence format)

    Returns:
        Callback function compatible with euler_denoising_loop
    """
    os.makedirs(preview_dir, exist_ok=True)

    # Track collected frames for video preview
    collected_frames = []

    # Determine MP4 output path
    suffix = f"_{preview_suffix}" if preview_suffix else ""
    mp4_path = os.path.join(preview_dir, f"latent_preview{suffix}.mp4")

    def callback(step_idx: int, video_state, sigmas: torch.Tensor):
        if step_idx % preview_interval != 0:
            return

        preview = generate_preview(
            latents=video_state.latent,
            max_height=max_height,
            num_keyframes=4,
            latent_frames=latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
        )

        if preview is not None:
            # Save individual frame (optional, useful for debugging)
            sigma = sigmas[step_idx].item() if step_idx < len(sigmas) else 0.0
            preview_path = os.path.join(
                preview_dir,
                f"{stage_name}_step_{step_idx:03d}_sigma_{sigma:.4f}.jpg",
            )
            preview.save(preview_path, quality=85)

            # Also save/update MP4 preview video (for UI monitoring)
            try:
                import numpy as np
                frame_array = np.array(preview)
                collected_frames.append(frame_array)

                # Write MP4 with all collected frames
                if len(collected_frames) > 0:
                    import imageio
                    # Use imageio to write MP4 (overwrites each time with all frames)
                    imageio.mimwrite(mp4_path, collected_frames, fps=2, quality=7)
                    print(f">>> Preview video updated: {mp4_path} ({len(collected_frames)} frames)")
            except ImportError:
                print(f">>> Preview saved: {preview_path} (imageio not available for MP4)")
            except Exception as e:
                print(f">>> Preview saved: {preview_path} (MP4 write error: {e})")

    return callback


# =============================================================================
# Video Continuation (Frame Freezing) Helpers
# =============================================================================

def prepare_frame_freezing(
    video_tensor: torch.Tensor,
    video_encoder,
    freeze_frames: int,
    num_frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare frozen latent and freeze mask for video continuation.

    Args:
        video_tensor: Input video frames [F, H, W, C] in 0-255 uint8
        video_encoder: LTX video encoder
        freeze_frames: Number of pixel frames to freeze
        num_frames: Total pixel frames in output video
        device: Target device
        dtype: Target dtype

    Returns:
        Tuple of (frozen_latent [1, C, lat_T, lat_H, lat_W], freeze_mask [1, 1, lat_T, lat_H, lat_W])
    """
    # Extract frames to freeze
    frames_to_freeze = video_tensor[:freeze_frames].float()
    if frames_to_freeze.max() > 1.0:
        frames_to_freeze = frames_to_freeze / 255.0

    # Convert from [F, H, W, C] to [1, C, F, H, W]
    frames_to_freeze = frames_to_freeze.permute(3, 0, 1, 2).unsqueeze(0)
    frames_to_freeze = frames_to_freeze.to(device=device, dtype=dtype)

    # Normalize to [-1, 1] for VAE
    frames_to_freeze = frames_to_freeze * 2.0 - 1.0

    # Encode to latent
    with torch.no_grad():
        frozen_latent = video_encoder(frames_to_freeze)  # [1, C, lat_F, lat_H, lat_W]

    # Calculate latent dimensions
    lat_F = frozen_latent.shape[2]
    lat_H = frozen_latent.shape[3]
    lat_W = frozen_latent.shape[4]

    # Total latent frames for the full video
    total_lat_frames = (num_frames - 1) // 8 + 1

    # Create freeze mask: 1 for frozen frames, 0 for generated frames
    freeze_mask = torch.zeros(1, 1, total_lat_frames, lat_H, lat_W, device=device, dtype=dtype)
    freeze_mask[:, :, :lat_F, :, :] = 1.0

    # Pad frozen_latent to full temporal length (zeros for non-frozen parts)
    full_frozen_latent = torch.zeros(
        1, frozen_latent.shape[1], total_lat_frames, lat_H, lat_W,
        device=device, dtype=dtype
    )
    full_frozen_latent[:, :, :lat_F, :, :] = frozen_latent

    return full_frozen_latent, freeze_mask


# =============================================================================
# Sliding Window (Long Video) Generation
# =============================================================================

def prepare_overlap_injection(
    prev_latent: torch.Tensor,
    overlap_noise: float,
) -> torch.Tensor:
    """
    Prepare overlapped latents with noise for sliding window injection.

    Wan2GP approach: blend previous latent with noise based on overlap_noise.

    Args:
        prev_latent: Latent from end of previous window [1, C, T, H, W]
        overlap_noise: Noise level (0-100 scale)

    Returns:
        Prepared latent tensor for injection
    """
    # Convert overlap_noise from 0-100 scale to 0-1 sigma scale
    noise_sigma = overlap_noise / 100.0

    if noise_sigma > 0:
        noise = torch.randn_like(prev_latent) * noise_sigma
        return prev_latent * (1 - noise_sigma) + noise

    return prev_latent


def apply_lab_color_correction(
    current_window: torch.Tensor,
    reference_window: torch.Tensor,
    strength: float = 1.0,
) -> torch.Tensor:
    """
    Apply LAB color space transfer for consistent colors across windows.

    Args:
        current_window: Current window pixels [F, H, W, C] in 0-255 range
        reference_window: Reference pixels from previous window [F_ref, H, W, C]
        strength: Blending strength (0=none, 1=full)

    Returns:
        Color-corrected current window
    """
    if strength == 0:
        return current_window

    try:
        from skimage import color
        import numpy as np

        # Convert to float [0, 1] for color space conversion
        current_np = current_window.float().numpy() / 255.0
        reference_np = reference_window.float().numpy() / 255.0

        # Compute LAB statistics for reference
        reference_lab = color.rgb2lab(reference_np)
        ref_mean = np.mean(reference_lab, axis=(0, 1, 2))
        ref_std = np.std(reference_lab, axis=(0, 1, 2)) + 1e-8

        # Apply correction to entire current window
        current_lab = color.rgb2lab(current_np)
        cur_mean = np.mean(current_lab, axis=(0, 1, 2))
        cur_std = np.std(current_lab, axis=(0, 1, 2)) + 1e-8

        # Transfer statistics: (x - cur_mean) * (ref_std / cur_std) + ref_mean
        corrected_lab = (current_lab - cur_mean) * (ref_std / cur_std) + ref_mean

        # Convert back to RGB
        corrected_rgb = color.lab2rgb(corrected_lab)
        corrected_rgb = np.clip(corrected_rgb * 255, 0, 255).astype(np.uint8)

        # Blend with original based on strength
        original = current_window.numpy()
        result = (corrected_rgb * strength + original * (1 - strength)).astype(np.uint8)
        return torch.from_numpy(result)

    except ImportError:
        print("Warning: skimage not available, skipping color correction")
        return current_window
    except Exception as e:
        print(f"Warning: Color correction failed: {e}")
        return current_window


def sliding_window_generate(
    generator: "LTXVideoGeneratorWithOffloading",
    args,
    total_frames: int,
    window_size: int = 129,
    overlap: int = 9,
    overlap_noise: float = 0.0,
    color_correction_strength: float = 0.0,
    latent_norm_fn: Callable | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Generate long videos using sliding window approach.

    Divides the video into overlapping windows, generates each window
    sequentially, and concatenates with proper blending.

    Args:
        generator: LTXVideoGeneratorWithOffloading instance
        args: Command line arguments
        total_frames: Total frames to generate
        window_size: Frames per window (should be 8n+1 for latent alignment)
        overlap: Overlapping frames between windows (should be 8n+1)
        overlap_noise: Noise level (0-100) for overlapped frames
        color_correction_strength: LAB color correction strength (0-1)

    Returns:
        Tuple of (video_tensor [F, H, W, C], audio_tensor or None)
    """
    from PIL import Image

    # Validate alignment to LTX's 8-frame temporal compression
    if (window_size - 1) % 8 != 0:
        print(f"Warning: window_size {window_size} not aligned to 8n+1, adjusting...")
        window_size = ((window_size - 1) // 8) * 8 + 1

    if overlap > 0 and (overlap - 1) % 8 != 0:
        print(f"Warning: overlap {overlap} not aligned to 8n+1, adjusting...")
        overlap = ((overlap - 1) // 8) * 8 + 1

    # Calculate number of windows needed
    effective_window = window_size - overlap
    if effective_window <= 0:
        raise ValueError(f"Overlap ({overlap}) must be less than window_size ({window_size})")

    num_windows = max(1, (total_frames - overlap + effective_window - 1) // effective_window)

    print(f">>> Sliding Window: {num_windows} windows of {window_size} frames")
    print(f">>> Overlap: {overlap} frames, Effective: {effective_window} new frames/window")

    all_video_chunks = []
    all_audio_chunks = []
    prev_window_latent = None
    prev_window_pixels = None

    device = generator.device
    dtype = generator.dtype

    for window_idx in range(num_windows):
        print("=" * 60)
        print(f">>> Sliding Window {window_idx + 1}/{num_windows}")
        print("=" * 60)

        # Calculate frame range for this window
        start_frame = window_idx * effective_window
        end_frame = min(start_frame + window_size, total_frames)
        window_frames = end_frame - start_frame

        # Ensure window_frames is valid (8n+1)
        if window_frames < 9:
            window_frames = 9
        if (window_frames - 1) % 8 != 0:
            window_frames = ((window_frames - 1) // 8 + 1) * 8 + 1

        # Use same seed for all windows to maintain coherence
        # The overlap latent conditioning handles continuity between windows
        window_seed = args.seed

        # Build conditionings for this window
        window_images = list(args.images) if args.images else []

        # Prepare overlapped latents injection (if not first window)
        overlap_latent = None
        num_overlap_latent = 0
        if window_idx > 0 and prev_window_latent is not None:
            # Inject previous window's ending latent as start conditioning
            overlap_latent = prepare_overlap_injection(prev_window_latent, overlap_noise)
            # Calculate number of latent frames from pixel overlap
            # LTX temporal compression is 8x, so overlap pixels / 8 ≈ latent frames
            num_overlap_latent = prev_window_latent.shape[2]  # Use actual latent frame count

        print(f">>> Window {window_idx + 1}: frames {start_frame}-{end_frame}, {window_frames} frames, seed {window_seed}")

        # Create preview callback if enabled
        preview_callback = None
        if args.preview_dir:
            # Compute latent dimensions for preview unpatchification
            # Two-stage operates at half resolution in stage 1
            if generator.one_stage:
                lf = (window_frames - 1) // 8 + 1
                lh = args.height // 32
                lw = args.width // 32
            else:
                lf = (window_frames - 1) // 8 + 1
                lh = (args.height // 2) // 32
                lw = (args.width // 2) // 32

            preview_callback = create_preview_callback(
                preview_dir=args.preview_dir,
                preview_interval=args.preview_interval,
                max_height=args.preview_max_height,
                stage_name=f"window{window_idx + 1}",
                preview_suffix=args.preview_suffix,
                latent_frames=lf,
                latent_height=lh,
                latent_width=lw,
            )

        # Generate this window
        video_iterator, audio, _ = generator.generate(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seed=window_seed,
            height=args.height,
            width=args.width,
            num_frames=window_frames,
            frame_rate=args.frame_rate,
            num_inference_steps=args.num_inference_steps,
            cfg_guidance_scale=args.cfg_guidance_scale,
            images=window_images,
            tiling_config=TilingConfig.default(),
            enhance_prompt=False,
            disable_audio=args.disable_audio,
            stage2_steps=args.stage2_steps,
            anchor_image=args.anchor_image,
            anchor_interval=args.anchor_interval,
            anchor_strength=args.anchor_strength,
            anchor_decay=args.anchor_decay,
            audio=args.audio,
            audio_strength=args.audio_strength,
            # STG parameters
            stg_scale=args.stg_scale,
            stg_blocks=args.stg_blocks,
            stg_mode=args.stg_mode,
            # Sliding window overlap conditioning
            _overlap_latent=overlap_latent,
            _num_overlap_latent=num_overlap_latent,
            _overlap_strength=0.95,
            preview_callback=preview_callback,
            preview_callback_interval=args.preview_interval,
            # Depth Control (IC-LoRA) parameters
            depth_video=args.depth_video,
            depth_image=args.depth_image,
            estimate_depth=args.estimate_depth,
            depth_strength=args.depth_strength,
            depth_stage2=args.depth_stage2,
            # Latent normalization
            latent_norm_fn=latent_norm_fn,
        )

        # Collect video frames from iterator - move to CPU immediately to save GPU memory
        video_frames = []
        for chunk in video_iterator:
            video_frames.append(chunk.cpu())  # Move to CPU immediately
            del chunk  # Free GPU memory
        # Concatenate on CPU (all downstream ops work on CPU: color correction uses numpy, overlap extraction uses cv2)
        video_tensor = torch.cat(video_frames, dim=0)  # [F, H, W, C] on CPU
        del video_frames  # Free list memory after concatenation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f">>> Window {window_idx + 1} generated: {video_tensor.shape}")

        # Extract ending latent for next window (if not last window)
        if window_idx < num_windows - 1:
            # Extract last overlap frames for latent encoding
            overlap_pixel_frames = overlap if overlap > 0 else 8
            frames_for_latent = video_tensor[-overlap_pixel_frames:, :, :, :]

            # Encode to latent
            video_encoder = generator.stage_1_model_ledger.video_encoder()

            # Resize if two-stage pipeline
            if not generator.one_stage:
                import cv2
                stage_1_h = args.height // 2
                stage_1_w = args.width // 2
                frames_resized = []
                for i in range(frames_for_latent.shape[0]):
                    frame = frames_for_latent[i].cpu().numpy().astype("uint8")
                    frame_resized = cv2.resize(frame, (stage_1_w, stage_1_h), interpolation=cv2.INTER_LANCZOS4)
                    frames_resized.append(torch.from_numpy(frame_resized).float())
                frames_for_latent = torch.stack(frames_resized, dim=0)

            prev_window_latent = _encode_frames_to_latent(
                frames_for_latent,
                video_encoder,
                device,
                dtype,
                target_latent_frames=(overlap_pixel_frames - 1) // 8 + 1,
            )
            print(f">>> Extracted overlap latent: {prev_window_latent.shape}")

            video_encoder.to("cpu")
            del video_encoder
            synchronize_and_cleanup()

        # Apply color correction if enabled and not first window
        if color_correction_strength > 0 and prev_window_pixels is not None:
            video_tensor = apply_lab_color_correction(
                current_window=video_tensor,
                reference_window=prev_window_pixels,
                strength=color_correction_strength,
            )

        # Store reference for next window's color correction
        prev_window_pixels = video_tensor[-overlap:] if overlap > 0 else video_tensor[-8:]

        # Store chunk (strip overlap from non-first windows)
        # video_tensor is already on CPU from earlier collection
        if window_idx == 0:
            all_video_chunks.append(video_tensor)
        else:
            all_video_chunks.append(video_tensor[overlap:].clone())

        del video_tensor

        if audio is not None:
            samples_per_frame = int(AUDIO_SAMPLE_RATE / args.frame_rate)
            if window_idx == 0:
                all_audio_chunks.append(audio)
            else:
                samples_to_skip = overlap * samples_per_frame
                # Audio is [channels, samples], so slice along dim=1
                all_audio_chunks.append(audio[:, samples_to_skip:])

    # Concatenate all windows
    print("=" * 60)
    print(">>> Sliding Window: Concatenating windows...")
    print("=" * 60)

    final_video = torch.cat(all_video_chunks, dim=0)
    # Trim to exact requested length
    if final_video.shape[0] > total_frames:
        final_video = final_video[:total_frames]

    # Audio is [channels, samples], concatenate along samples dimension (dim=1)
    final_audio = torch.cat(all_audio_chunks, dim=1) if all_audio_chunks else None

    print(f">>> Final video shape: {final_video.shape}")
    if final_audio is not None:
        print(f">>> Final audio shape: {final_audio.shape}")

    return final_video, final_audio


# =============================================================================
# Main Entry Point
# =============================================================================

def add_metadata_to_video(video_path: str, parameters: dict) -> None:
    """Add generation parameters to video metadata using ffmpeg."""
    import json
    import subprocess

    params_json = json.dumps(parameters, indent=2)
    temp_path = video_path.replace(".mp4", "_temp.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-metadata", f"comment={params_json}",
        "-codec", "copy",
        temp_path
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.replace(temp_path, video_path)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to add metadata: {e.stderr.decode() if e.stderr else e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as e:
        print(f"Warning: Failed to add metadata: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)


@torch.inference_mode()
def main():
    """Main entry point for LTX-2 video generation."""
    args = parse_args()

    # V2A mode validation
    if args.v2a_mode:
        if not args.input_video:
            print("Error: --v2a-mode requires --input-video")
            sys.exit(1)
        if args.disable_audio:
            print(">>> Warning: --disable-audio is ignored in V2A mode")
            args.disable_audio = False
        if args.refine_only:
            print("Error: --v2a-mode cannot be combined with --refine-only")
            sys.exit(1)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s")

    # Handle default distilled LoRA
    # Don't auto-add distilled LoRA if:
    # - User explicitly set --distilled-lora (even to empty)
    # - Using a distilled checkpoint (checkpoint itself is distilled)
    # - Using a separate stage 2 checkpoint (assumed to be a complete model)
    if args.distilled_lora is None:
        if args.distilled_checkpoint or args.stage2_checkpoint:
            args.distilled_lora = []  # No distilled LoRA needed
        else:
            args.distilled_lora = [LoraPathStrengthAndSDOps(
                resolve_path(DEFAULT_DISTILLED_LORA_PATH),
                DEFAULT_LORA_STRENGTH,
                LTXV_LORA_COMFY_RENAMING_MAP
            )]

    pipeline_type = "refine-only" if args.refine_only else ("one-stage" if args.one_stage else "two-stage")
    checkpoint_type = "distilled" if args.distilled_checkpoint else "standard"
    print("=" * 60)
    print("LTX-2 Video Generation")
    print("=" * 60)
    print(f"Pipeline: {pipeline_type} ({checkpoint_type} checkpoint)")
    print(f"Checkpoint: {args.checkpoint_path}")
    if args.stage2_checkpoint:
        print(f"Stage 2 Checkpoint: {args.stage2_checkpoint}")
    if not args.one_stage:
        print(f"Stage 2 Steps: {args.stage2_steps}")
    print(f"Output: {args.output_path}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Frames: {args.num_frames} ({args.num_frames / args.frame_rate:.1f}s at {args.frame_rate}fps)")
    print(f"Steps: {args.num_inference_steps}, CFG: {args.cfg_guidance_scale}, STG: {args.stg_scale}")
    if args.stg_scale > 0:
        print(f"STG Blocks: {args.stg_blocks}, Mode: {args.stg_mode}")
    print(f"Seed: {args.seed}")
    print(f"Offload: {args.offload}")
    print(f"FP8: {args.enable_fp8}")
    if args.images:
        print(f"Image conditioning: {len(args.images)} image(s)")
    if args.anchor_interval:
        anchor_src = args.anchor_image if args.anchor_image else "first --image"
        num_anchors = len(range(args.anchor_interval, args.num_frames, args.anchor_interval))
        decay_info = f", decay: {args.anchor_decay}" if args.anchor_decay != "none" else ""
        print(f"Anchor conditioning: {num_anchors} anchor(s) every {args.anchor_interval} frames (source: {anchor_src}, strength: {args.anchor_strength}{decay_info})")
    # SVI Pro mode info
    if args.svi_mode or args.extend_video:
        svi_mode_type = "Video Extension" if args.extend_video else "Multi-Clip Generation"
        print(f"SVI Pro Mode: {svi_mode_type}")
        print(f"  Clips: {args.num_clips}")
        print(f"  Motion Latent Frames: {args.num_motion_latent}")
        print(f"  Seed Multiplier: {args.seed_multiplier}")
        if args.extend_video:
            print(f"  Input Video: {args.extend_video}")
            print(f"  Prepend Original: {args.prepend_original}")
    # V2A mode info
    if args.v2a_mode:
        print(f"V2A Mode: Video-to-Audio (freeze video, generate audio)")
        print(f"  Input Video: {args.input_video}")
    # AV Extension mode info
    if args.av_extend_from:
        print(f"AV Extension Mode: Time-Based Audio-Video Continuation")
        print(f"  Input Video: {args.av_extend_from}")
        print(f"  Start Time: {args.av_extend_start_time or 'auto (end of video)'}s")
        print(f"  End Time: {args.av_extend_end_time or 'auto (start + 5s)'}s")
        print(f"  Extension Steps: {args.av_extend_steps}")
        print(f"  Terminal Sigma: {args.av_extend_terminal}")
        print(f"  Skip Stage 2: {args.av_no_stage2}")
    print("=" * 60)
    print(f"Prompt: {args.prompt}")
    print("=" * 60)

    # Create generator
    generator = LTXVideoGeneratorWithOffloading(
        checkpoint_path=args.checkpoint_path,
        distilled_lora=args.distilled_lora,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=args.loras,
        stage2_loras=args.stage2_loras,
        fp8transformer=args.enable_fp8,
        offload=args.offload,
        enable_dit_block_swap=args.enable_dit_block_swap,
        dit_blocks_in_memory=args.dit_blocks_in_memory,
        enable_text_encoder_block_swap=args.enable_text_encoder_block_swap,
        text_encoder_blocks_in_memory=args.text_encoder_blocks_in_memory,
        enable_refiner_block_swap=args.enable_refiner_block_swap,
        refiner_blocks_in_memory=args.refiner_blocks_in_memory,
        enable_activation_offload=args.enable_activation_offload,
        temporal_chunk_size=args.temporal_chunk_size,
        one_stage=args.one_stage,
        refine_only=args.refine_only,
        distilled_checkpoint=args.distilled_checkpoint,
        stage2_checkpoint=args.stage2_checkpoint,
        ffn_chunk_size=args.ffn_chunk_size,
        vae_path=args.vae,
    )

    # Set up tiling config for VAE
    tiling_config = TilingConfig.default()

    # Create latent normalization function from args (fixes overbaking/audio clipping)
    latent_norm_fn = None
    if args.latent_norm != "none":
        apply_to_video = not args.latent_norm_audio_only
        apply_to_audio = not args.latent_norm_video_only
        if args.latent_norm == "stat":
            latent_norm_fn = create_per_step_stat_norm_fn(
                factors=args.latent_norm_factors,
                target_mean=args.latent_norm_target_mean,
                target_std=args.latent_norm_target_std,
                percentile=args.latent_norm_percentile,
                clip_outliers=args.latent_norm_clip_outliers,
                apply_to_video=apply_to_video,
                apply_to_audio=apply_to_audio,
            )
            print(f">>> Latent normalization: statistical (factors: {args.latent_norm_factors})")
        elif args.latent_norm == "adain":
            # For AdaIN, we need a reference latent - this would require encoding a reference image/video
            # For now, fall back to stat norm if adain is requested without reference
            print(">>> Warning: AdaIN normalization requires reference latent, falling back to statistical norm")
            latent_norm_fn = create_per_step_stat_norm_fn(
                factors=args.latent_norm_factors,
                target_mean=args.latent_norm_target_mean,
                target_std=args.latent_norm_target_std,
                percentile=args.latent_norm_percentile,
                clip_outliers=args.latent_norm_clip_outliers,
                apply_to_video=apply_to_video,
                apply_to_audio=apply_to_audio,
            )

    # Determine if sliding window mode should be used (requires explicit flag)
    use_sliding_window = (
        args.enable_sliding_window
        and args.num_frames > args.sliding_window_size
        and not args.svi_mode
        and not args.extend_video
        and not args.av_extend_from
    )

    # Track enhanced prompt for metadata (only set in regular generation mode)
    enhanced_prompt = None

    # Branch between sliding window, SVI Pro mode, and regular mode
    if use_sliding_window:
        # Sliding window mode for long videos
        print("=" * 60)
        print(f">>> Using Sliding Window mode: {args.num_frames} frames > {args.sliding_window_size} window size")
        print("=" * 60)

        video_tensor, audio = sliding_window_generate(
            generator=generator,
            args=args,
            total_frames=args.num_frames,
            window_size=args.sliding_window_size,
            overlap=args.sliding_window_overlap,
            overlap_noise=args.sliding_window_overlap_noise,
            color_correction_strength=args.sliding_window_color_correction,
            latent_norm_fn=latent_norm_fn,
        )

        # Convert to iterator for encode_video
        def tensor_to_iterator(tensor):
            yield tensor

        video = tensor_to_iterator(video_tensor)
        total_frames = video_tensor.shape[0]
        video_chunks_number = get_video_chunks_number(total_frames, tiling_config)

    elif args.av_extend_from:
        # AV Extension mode: time-based audio-video masking
        print("=" * 60)
        print(">>> Using AV Extension mode (time-based audio-video masking)")
        print("=" * 60)

        video_tensor, audio = generate_av_extension(
            generator=generator,
            args=args,
            input_video_path=args.av_extend_from,
            start_time=args.av_extend_start_time,
            end_time=args.av_extend_end_time,
            extend_steps=args.av_extend_steps,
            terminal=args.av_extend_terminal,
            slope_len=args.av_slope_len,
            skip_stage2=args.av_no_stage2,
            latent_norm_fn=latent_norm_fn,
        )

        # Convert to iterator for encode_video
        def tensor_to_iterator(tensor):
            yield tensor

        video = tensor_to_iterator(video_tensor)
        total_frames = video_tensor.shape[0]
        video_chunks_number = get_video_chunks_number(total_frames, tiling_config)

    elif args.v2v_join_video1 and args.v2v_join_video2:
        # V2V Join mode: join two videos with generated transition
        print("=" * 60)
        print(">>> Using V2V Join mode (video-to-video transition)")
        print("=" * 60)

        video_tensor, audio = generate_v2v_join(
            generator=generator,
            args=args,
            video1_path=args.v2v_join_video1,
            video2_path=args.v2v_join_video2,
            frames_check1=args.v2v_join_frames_check1,
            frames_check2=args.v2v_join_frames_check2,
            preserve1_sec=args.v2v_join_preserve1,
            preserve2_sec=args.v2v_join_preserve2,
            transition_sec=args.v2v_join_transition_time,
            extend_steps=args.v2v_join_steps,
            terminal=args.v2v_join_terminal,
            slope_len=args.av_slope_len,
            skip_stage2=args.av_no_stage2,
            latent_norm_fn=latent_norm_fn,
        )

        # Convert to iterator for encode_video
        def tensor_to_iterator(tensor):
            yield tensor

        video = tensor_to_iterator(video_tensor)
        total_frames = video_tensor.shape[0]
        video_chunks_number = get_video_chunks_number(total_frames, tiling_config)

    elif args.svi_mode or args.extend_video:
        # SVI Pro mode: multi-clip or video extension
        # Parse prompt list if provided
        prompts = None
        if getattr(args, 'prompt_list', None) is not None and len(args.prompt_list) > 0:
            prompts = args.prompt_list
            print(f">>> Using {len(prompts)} prompts for multi-clip generation")
            for i, p in enumerate(prompts):
                print(f">>>   Clip {i+1}: {p[:50]}..." if len(p) > 50 else f">>>   Clip {i+1}: {p}")
        else:
            print(f">>> Using single prompt for all clips: {args.prompt}")

        if args.extend_video:
            # Video extension mode
            if not args.images:
                # No explicit image provided, will use frames from video
                args.images = []
            video_tensor, audio = generate_svi_video_extension(
                generator=generator,
                args=args,
                input_video_path=args.extend_video,
                num_clips=args.num_clips,
                num_motion_latent=args.num_motion_latent,
                num_motion_frame=args.num_motion_frame,
                seed_multiplier=args.seed_multiplier,
                overlap_frames=args.overlap_frames,
                anchor_image_path=args.anchor_image,
                prepend_original=args.prepend_original,
                prompts=prompts,
                latent_norm_fn=latent_norm_fn,
            )
        else:
            # Multi-clip generation mode
            if not args.images:
                raise ValueError("SVI mode requires at least one --image for the initial frame")
            initial_image = args.images[0][0]
            video_tensor, audio = generate_svi_multi_clip(
                generator=generator,
                args=args,
                initial_image_path=initial_image,
                num_clips=args.num_clips,
                num_motion_latent=args.num_motion_latent,
                num_motion_frame=args.num_motion_frame,
                seed_multiplier=args.seed_multiplier,
                overlap_frames=args.overlap_frames,
                prompts=prompts,
                latent_norm_fn=latent_norm_fn,
            )

        # SVI returns a decoded tensor [F, H, W, C], convert to iterator format for encode_video
        def tensor_to_iterator(tensor):
            """Convert tensor to chunk iterator for encode_video compatibility."""
            yield tensor

        video = tensor_to_iterator(video_tensor)
        # Calculate total frames for proper video chunk handling
        total_frames = video_tensor.shape[0]
        video_chunks_number = get_video_chunks_number(total_frames, tiling_config)
    else:
        # Regular single-clip generation
        video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)

        # Create preview callback if enabled
        preview_callback = None
        if args.preview_dir:
            # Compute latent dimensions for preview unpatchification
            # Two-stage operates at half resolution in stage 1
            if generator.one_stage:
                lf = (args.num_frames - 1) // 8 + 1
                lh = args.height // 32
                lw = args.width // 32
            else:
                lf = (args.num_frames - 1) // 8 + 1
                lh = (args.height // 2) // 32
                lw = (args.width // 2) // 32

            preview_callback = create_preview_callback(
                preview_dir=args.preview_dir,
                preview_interval=args.preview_interval,
                max_height=args.preview_max_height,
                stage_name="stage1",
                preview_suffix=args.preview_suffix,
                latent_frames=lf,
                latent_height=lh,
                latent_width=lw,
            )

        video, audio, enhanced_prompt = generator.generate(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            num_inference_steps=args.num_inference_steps,
            cfg_guidance_scale=args.cfg_guidance_scale,
            images=args.images,
            tiling_config=tiling_config,
            enhance_prompt=args.enhance_prompt,
            disable_audio=args.disable_audio,
            input_video=args.input_video,
            refine_strength=args.refine_strength,
            refine_steps=args.refine_steps,
            stage2_steps=args.stage2_steps,
            anchor_image=args.anchor_image,
            anchor_interval=args.anchor_interval,
            anchor_strength=args.anchor_strength,
            anchor_decay=args.anchor_decay,
            audio=args.audio,
            audio_strength=args.audio_strength,
            # STG parameters
            stg_scale=args.stg_scale,
            stg_blocks=args.stg_blocks,
            stg_mode=args.stg_mode,
            preview_callback=preview_callback,
            preview_callback_interval=args.preview_interval,
            # Depth Control (IC-LoRA) parameters
            depth_video=args.depth_video,
            depth_image=args.depth_image,
            estimate_depth=args.estimate_depth,
            depth_strength=args.depth_strength,
            depth_stage2=args.depth_stage2,
            # Latent normalization (fixes overbaking/audio clipping)
            latent_norm_fn=latent_norm_fn,
            # V2A mode (freeze video, generate audio)
            v2a_mode=args.v2a_mode,
        )

    # Encode and save video
    print(f">>> Encoding video to {args.output_path}...")
    encode_start = time.time()

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE if audio is not None else None,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )

    print(f">>> Video saved in {time.time() - encode_start:.1f}s")

    # Build and save metadata
    metadata = {
        "model_type": "LTX-2",
        "pipeline": "refine-only" if args.refine_only else ("one-stage" if args.one_stage else "two-stage"),
        "distilled_checkpoint": args.distilled_checkpoint,
        "checkpoint_path": args.checkpoint_path,
        "stage2_checkpoint": args.stage2_checkpoint,
        "stage2_steps": args.stage2_steps if not args.one_stage else None,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt if args.cfg_guidance_scale > 1.0 else None,
        "width": args.width,
        "height": args.height,
        "num_frames": args.num_frames,
        "frame_rate": args.frame_rate,
        "cfg_guidance_scale": args.cfg_guidance_scale,
        "stg_scale": args.stg_scale,
        "stg_blocks": args.stg_blocks if args.stg_scale > 0 else None,
        "stg_mode": args.stg_mode if args.stg_scale > 0 else None,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "offload": args.offload,
        "enable_fp8": args.enable_fp8,
        # Separate block swap settings
        "enable_dit_block_swap": args.enable_dit_block_swap,
        "dit_blocks_in_memory": args.dit_blocks_in_memory if args.enable_dit_block_swap else None,
        "enable_text_encoder_block_swap": args.enable_text_encoder_block_swap,
        "text_encoder_blocks_in_memory": args.text_encoder_blocks_in_memory if args.enable_text_encoder_block_swap else None,
        "enable_refiner_block_swap": args.enable_refiner_block_swap,
        "refiner_blocks_in_memory": args.refiner_blocks_in_memory if args.enable_refiner_block_swap else None,
        "images": [(img[0], img[1], img[2]) for img in args.images] if args.images else None,
        "loras": [(lora.path, lora.strength) for lora in args.loras] if args.loras else None,
        "distilled_lora": [(lora.path, lora.strength) for lora in args.distilled_lora] if args.distilled_lora else None,
        "input_video": args.input_video,
        "refine_strength": args.refine_strength if args.input_video else None,
        "refine_steps": args.refine_steps if args.input_video else None,
        "anchor_image": args.anchor_image,
        "anchor_interval": args.anchor_interval,
        "anchor_strength": args.anchor_strength if args.anchor_interval else None,
        "anchor_decay": args.anchor_decay if args.anchor_interval and args.anchor_decay != "none" else None,
        "disable_audio": args.disable_audio,
        "audio": args.audio,
        "audio_strength": args.audio_strength if args.audio else None,
        "enhance_prompt": args.enhance_prompt,
        "enhanced_prompt": enhanced_prompt,
        # SVI Pro metadata
        "svi_mode": args.svi_mode,
        "svi_num_clips": args.num_clips if (args.svi_mode or args.extend_video) else None,
        "svi_num_motion_latent": args.num_motion_latent if (args.svi_mode or args.extend_video) else None,
        "svi_num_motion_frame": args.num_motion_frame if (args.svi_mode or args.extend_video) else None,
        "svi_seed_multiplier": args.seed_multiplier if (args.svi_mode or args.extend_video) else None,
        "svi_overlap_frames": args.overlap_frames if (args.svi_mode or args.extend_video) else None,
        "svi_extend_video": args.extend_video,
        "svi_prepend_original": args.prepend_original if args.extend_video else None,
        # Sliding window metadata
        "sliding_window_mode": use_sliding_window,
        "sliding_window_size": args.sliding_window_size if use_sliding_window else None,
        "sliding_window_overlap": args.sliding_window_overlap if use_sliding_window else None,
        "sliding_window_overlap_noise": args.sliding_window_overlap_noise if use_sliding_window else None,
        "sliding_window_color_correction": args.sliding_window_color_correction if use_sliding_window else None,
        # Preview metadata
        "preview_dir": args.preview_dir,
        "preview_interval": args.preview_interval if args.preview_dir else None,
        # Depth Control (IC-LoRA) metadata
        "depth_video": args.depth_video,
        "depth_image": args.depth_image,
        "estimate_depth": args.estimate_depth,
        "depth_strength": args.depth_strength if (args.depth_video or args.depth_image or args.estimate_depth) else None,
        "depth_stage2": args.depth_stage2 if (args.depth_video or args.depth_image or args.estimate_depth) else None,
        # Video continuation metadata
        "freeze_frames": args.freeze_frames if args.freeze_frames > 0 else None,
        "freeze_transition": args.freeze_transition if args.freeze_frames > 0 else None,
    }

    print(">>> Adding metadata to video...")
    add_metadata_to_video(args.output_path, metadata)

    print(f">>> Output: {args.output_path}")
    print(">>> Done!")


if __name__ == "__main__":
    main()
