import gc
import logging
import math
from dataclasses import replace

import torch
from tqdm import tqdm

from ltx_core.components.noisers import Noiser
from ltx_core.components.protocols import DiffusionStepProtocol, GuiderProtocol
from ltx_core.conditioning import (
    ConditioningItem,
    VideoConditionByKeyframeIndex,
    VideoConditionByLatentIndex,
)
from ltx_core.model.transformer import Modality, X0Model
from ltx_core.model.video_vae import VideoEncoder
from ltx_core.text_encoders.gemma import GemmaTextEncoderModelBase
from ltx_core.tools import AudioLatentTools, LatentTools, VideoLatentTools
from ltx_core.types import AudioLatentShape, LatentState, VideoLatentShape, VideoPixelShape
from ltx_core.utils import to_denoised, to_velocity
from ltx_pipelines.utils.media_io import decode_image, load_image_conditioning, resize_aspect_ratio_preserving
from ltx_pipelines.utils.types import (
    DenoisingFunc,
    DenoisingLoopFunc,
    PipelineComponents,
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cleanup_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def image_conditionings_by_replacing_latent(
    images: list[tuple[str, int, float]],
    height: int,
    width: int,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
) -> list[ConditioningItem]:
    conditionings = []
    for image_path, frame_idx, strength in images:
        image = load_image_conditioning(
            image_path=image_path,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
        )
        encoded_image = video_encoder(image)
        conditionings.append(
            VideoConditionByLatentIndex(
                latent=encoded_image,
                strength=strength,
                latent_idx=frame_idx // 8,
            )
        )

    return conditionings


def image_conditionings_by_adding_guiding_latent(
    images: list[tuple[str, int, float]],
    height: int,
    width: int,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
) -> list[ConditioningItem]:
    conditionings = []
    for image_path, frame_idx, strength in images:
        image = load_image_conditioning(
            image_path=image_path,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
        )
        encoded_image = video_encoder(image)
        conditionings.append(
            VideoConditionByKeyframeIndex(keyframes=encoded_image, frame_idx=frame_idx, strength=strength)
        )
    return conditionings


def anchor_strength_decay(
    step_idx: int,
    total_steps: int,
    decay_type: str = "cosine",
) -> float:
    """
    Compute decay factor for anchor strength.

    Returns a value from 0.0 (keep original anchor constraint) to 1.0 (full freedom).
    This allows anchor constraints to be strong early in denoising (structure formation)
    and weak later (motion freedom).

    Args:
        step_idx: Current step index (0-based)
        total_steps: Total number of denoising steps
        decay_type: Decay curve type - "linear", "cosine", or "sigmoid"

    Returns:
        Decay factor in [0, 1] where 0=keep constraint, 1=release to full freedom
    """
    progress = step_idx / max(total_steps - 1, 1)
    if decay_type == "linear":
        return progress
    elif decay_type == "cosine":
        return (1 - math.cos(math.pi * progress)) / 2
    elif decay_type == "sigmoid":
        return 1 / (1 + math.exp(-10 * (progress - 0.5)))
    return progress  # Default to linear


def apply_frame_freezing(
    denoised_video: torch.Tensor,
    frozen_latent: torch.Tensor,
    freeze_mask: torch.Tensor,
    sigma: float,
    step_idx: int,
    total_steps: int,
    transition_steps: int = 4,
) -> torch.Tensor:
    """
    Apply frame freezing with noise-scaled blending at transition boundary.

    For frozen frames, inject source latent scaled by (1 - sigma) to maintain
    the original content while respecting the diffusion process.

    Args:
        denoised_video: Current denoised output [B, C, T, H, W]
        frozen_latent: Clean source latent for frozen frames [B, C, T, H, W]
        freeze_mask: Binary mask [B, 1, T, H, W], 1=frozen, 0=generated
        sigma: Current noise level
        step_idx: Current step index
        total_steps: Total denoising steps
        transition_steps: Steps for blending transition (unused, kept for API)

    Returns:
        Modified denoised video with frozen frames injected
    """
    # Expand freeze_mask to match latent channels
    expanded_mask = freeze_mask.expand_as(denoised_video)

    # For frozen frames: use clean latent scaled by how "done" we are
    # At high sigma (early): allow more denoising freedom
    # At low sigma (late): lock to clean latent
    sigma_max = 1.0
    noise_weight = min(sigma / sigma_max, 1.0)
    clean_weight = 1.0 - noise_weight

    # Blend: where mask=1, inject frozen content; where mask=0, keep denoised
    frozen_contribution = frozen_latent * clean_weight + denoised_video * noise_weight
    result = denoised_video * (1 - expanded_mask) + frozen_contribution * expanded_mask

    return result


def euler_denoising_loop(
    sigmas: torch.Tensor,
    video_state: LatentState,
    audio_state: LatentState,
    stepper: DiffusionStepProtocol,
    denoise_fn: DenoisingFunc,
    anchor_decay: str | None = None,
    callback: callable = None,
    callback_interval: int = 1,
    frozen_video_latent: torch.Tensor | None = None,
    freeze_mask: torch.Tensor | None = None,
    freeze_transition_steps: int = 4,
    latent_norm_fn: callable = None,
) -> tuple[LatentState, LatentState]:
    """
    Perform the joint audio-video denoising loop over a diffusion schedule.
    This function iterates over all but the final value in ``sigmas`` and, at
    each diffusion step, calls ``denoise_fn`` to obtain denoised video and
    audio latents. The denoised latents are post-processed with their
    respective denoise masks and clean latents, then passed to ``stepper`` to
    advance the noisy latents one step along the diffusion schedule.
    ### Parameters
    sigmas:
        A 1D tensor of noise levels (diffusion sigmas) defining the sampling
        schedule. All steps except the last element are iterated over.
    video_state:
        The current video :class:`LatentState`, containing the noisy latent,
        its clean reference latent, and the denoising mask.
    audio_state:
        The current audio :class:`LatentState`, analogous to ``video_state``
        but for the audio modality.
    stepper:
        An implementation of :class:`DiffusionStepProtocol` that updates a
        latent given the current latent, its denoised estimate, the full
        ``sigmas`` schedule, and the current step index.
    denoise_fn:
        A callable implementing :class:`DenoisingFunc`. It is invoked as
        ``denoise_fn(video_state, audio_state, sigmas, step_index)`` and must
        return a tuple ``(denoised_video, denoised_audio)``, where each element
        is a tensor with the same shape as the corresponding latent.
    anchor_decay:
        Optional decay schedule for anchor constraints. If provided, anchor tokens
        (those with denoise_mask < 1.0) will have their constraints gradually
        released over the denoising process. Options: "linear", "cosine", "sigmoid".
        This allows anchors to guide structure early but permit motion later.
    callback:
        Optional callback invoked at each step for preview generation.
        Signature: callback(step_idx, video_state, sigmas)
    callback_interval:
        Invoke callback every N steps (default: 1 = every step).
    frozen_video_latent:
        If provided, these latents are injected where freeze_mask=1.
        Enables frame freezing for video continuation.
    freeze_mask:
        Tensor of shape [B, 1, T, H, W] where 1=frozen frame, 0=generated frame.
        Used with frozen_video_latent for selective frame injection.
    freeze_transition_steps:
        Number of steps for blending transition at freeze boundary.
    latent_norm_fn:
        Optional normalization function applied after each denoising step.
        Signature: fn(video_latent, audio_latent, step_idx) -> (video_latent, audio_latent)
        Use create_per_step_stat_norm_fn() or create_per_step_adain_norm_fn() to create.
        This helps fix overbaking and audio clipping by keeping latent statistics
        within expected bounds during denoising.
    ### Returns
    tuple[LatentState, LatentState]
        A pair ``(video_state, audio_state)`` containing the final video and
        audio latent states after completing the denoising loop.
    """
    total_steps = len(sigmas) - 1

    # Store original masks for anchor decay computation
    if anchor_decay:
        original_video_mask = video_state.denoise_mask.clone()

    for step_idx, _ in enumerate(tqdm(sigmas[:-1])):
        # Apply anchor decay: gradually release anchor constraints over time
        if anchor_decay:
            decay = anchor_strength_decay(step_idx, total_steps, anchor_decay)
            # Adjust mask: move toward 1.0 (full denoising freedom) based on decay
            # effective_mask = original_mask + (1.0 - original_mask) * decay
            adjusted_video_mask = original_video_mask + (1.0 - original_video_mask) * decay
            video_state = replace(video_state, denoise_mask=adjusted_video_mask)

        denoised_video, denoised_audio = denoise_fn(video_state, audio_state, sigmas, step_idx)

        # Apply latent normalization if enabled (fixes overbaking/audio clipping)
        if latent_norm_fn is not None:
            denoised_video, denoised_audio = latent_norm_fn(denoised_video, denoised_audio, step_idx)

        # Apply frame freezing if enabled
        if frozen_video_latent is not None and freeze_mask is not None:
            denoised_video = apply_frame_freezing(
                denoised_video=denoised_video,
                frozen_latent=frozen_video_latent,
                freeze_mask=freeze_mask,
                sigma=sigmas[step_idx].item(),
                step_idx=step_idx,
                total_steps=total_steps,
                transition_steps=freeze_transition_steps,
            )

        denoised_video = post_process_latent(denoised_video, video_state.denoise_mask, video_state.clean_latent)
        denoised_audio = post_process_latent(denoised_audio, audio_state.denoise_mask, audio_state.clean_latent)

        video_state = replace(video_state, latent=stepper.step(video_state.latent, denoised_video, sigmas, step_idx))
        audio_state = replace(audio_state, latent=stepper.step(audio_state.latent, denoised_audio, sigmas, step_idx))

        # Invoke callback for preview generation
        if callback is not None and step_idx % callback_interval == 0:
            callback(step_idx, video_state, sigmas)

    return (video_state, audio_state)


def gradient_estimating_euler_denoising_loop(
    sigmas: torch.Tensor,
    video_state: LatentState,
    audio_state: LatentState,
    stepper: DiffusionStepProtocol,
    denoise_fn: DenoisingFunc,
    ge_gamma: float = 2.0,
    anchor_decay: str | None = None,
    latent_norm_fn: callable = None,
) -> tuple[LatentState, LatentState]:
    """
    Perform the joint audio-video denoising loop using gradient-estimation sampling.
    This function is similar to :func:`euler_denoising_loop`, but applies
    gradient estimation to improve the denoised estimates by tracking velocity
    changes across steps. See the referenced function for detailed parameter
    documentation.
    ### Parameters
    ge_gamma:
        Gradient estimation coefficient controlling the velocity correction term.
        Default is 2.0. Paper: https://openreview.net/pdf?id=o2ND9v0CeK
    sigmas, video_state, audio_state, stepper, denoise_fn:
        See :func:`euler_denoising_loop` for parameter descriptions.
    anchor_decay:
        Optional decay schedule for anchor constraints. See :func:`euler_denoising_loop`.
    latent_norm_fn:
        Optional normalization function. See :func:`euler_denoising_loop`.
    ### Returns
    tuple[LatentState, LatentState]
        See :func:`euler_denoising_loop` for return value description.
    """
    total_steps = len(sigmas) - 1

    # Store original masks for anchor decay computation
    if anchor_decay:
        original_video_mask = video_state.denoise_mask.clone()

    previous_audio_velocity = None
    previous_video_velocity = None

    def update_velocity_and_sample(
        noisy_sample: torch.Tensor, denoised_sample: torch.Tensor, sigma: float, previous_velocity: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_velocity = to_velocity(noisy_sample, sigma, denoised_sample)
        if previous_velocity is not None:
            delta_v = current_velocity - previous_velocity
            total_velocity = ge_gamma * delta_v + previous_velocity
            denoised_sample = to_denoised(noisy_sample, total_velocity, sigma)
        return current_velocity, denoised_sample

    for step_idx, _ in enumerate(tqdm(sigmas[:-1])):
        # Apply anchor decay: gradually release anchor constraints over time
        if anchor_decay:
            decay = anchor_strength_decay(step_idx, total_steps, anchor_decay)
            adjusted_video_mask = original_video_mask + (1.0 - original_video_mask) * decay
            video_state = replace(video_state, denoise_mask=adjusted_video_mask)

        denoised_video, denoised_audio = denoise_fn(video_state, audio_state, sigmas, step_idx)

        # Apply latent normalization if enabled (fixes overbaking/audio clipping)
        if latent_norm_fn is not None:
            denoised_video, denoised_audio = latent_norm_fn(denoised_video, denoised_audio, step_idx)

        denoised_video = post_process_latent(denoised_video, video_state.denoise_mask, video_state.clean_latent)
        denoised_audio = post_process_latent(denoised_audio, audio_state.denoise_mask, audio_state.clean_latent)

        if sigmas[step_idx + 1] == 0:
            return replace(video_state, latent=denoised_video), replace(audio_state, latent=denoised_audio)

        previous_video_velocity, denoised_video = update_velocity_and_sample(
            video_state.latent, denoised_video, sigmas[step_idx], previous_video_velocity
        )
        previous_audio_velocity, denoised_audio = update_velocity_and_sample(
            audio_state.latent, denoised_audio, sigmas[step_idx], previous_audio_velocity
        )

        video_state = replace(video_state, latent=stepper.step(video_state.latent, denoised_video, sigmas, step_idx))
        audio_state = replace(audio_state, latent=stepper.step(audio_state.latent, denoised_audio, sigmas, step_idx))

    return (video_state, audio_state)


def noise_video_state(
    output_shape: VideoPixelShape,
    noiser: Noiser,
    conditionings: list[ConditioningItem],
    components: PipelineComponents,
    dtype: torch.dtype,
    device: torch.device,
    noise_scale: float = 1.0,
    initial_latent: torch.Tensor | None = None,
) -> tuple[LatentState, VideoLatentTools]:
    """Initialize and noise a video latent state for the diffusion pipeline.
    Creates a video latent state from the output shape, applies conditionings,
    and adds noise using the provided noiser. Returns the noised state and
    video latent tools for further processing. If initial_latent is provided, it will be used to create the initial
    state, otherwise an empty initial state will be created.
    """
    video_latent_shape = VideoLatentShape.from_pixel_shape(
        shape=output_shape,
        latent_channels=components.video_latent_channels,
        scale_factors=components.video_scale_factors,
    )
    video_tools = VideoLatentTools(components.video_patchifier, video_latent_shape, output_shape.fps)
    video_state = create_noised_state(
        tools=video_tools,
        conditionings=conditionings,
        noiser=noiser,
        dtype=dtype,
        device=device,
        noise_scale=noise_scale,
        initial_latent=initial_latent,
    )

    return video_state, video_tools


def noise_audio_state(
    output_shape: VideoPixelShape,
    noiser: Noiser,
    conditionings: list[ConditioningItem],
    components: PipelineComponents,
    dtype: torch.dtype,
    device: torch.device,
    noise_scale: float = 1.0,
    initial_latent: torch.Tensor | None = None,
) -> tuple[LatentState, AudioLatentTools]:
    """Initialize and noise an audio latent state for the diffusion pipeline.
    Creates an audio latent state from the output shape, applies conditionings,
    and adds noise using the provided noiser. Returns the noised state and
    audio latent tools for further processing. If initial_latent is provided, it will be used to create the initial
    state, otherwise an empty initial state will be created.
    """
    audio_latent_shape = AudioLatentShape.from_video_pixel_shape(output_shape)
    audio_tools = AudioLatentTools(components.audio_patchifier, audio_latent_shape)
    audio_state = create_noised_state(
        tools=audio_tools,
        conditionings=conditionings,
        noiser=noiser,
        dtype=dtype,
        device=device,
        noise_scale=noise_scale,
        initial_latent=initial_latent,
    )

    return audio_state, audio_tools


def create_noised_state(
    tools: LatentTools,
    conditionings: list[ConditioningItem],
    noiser: Noiser,
    dtype: torch.dtype,
    device: torch.device,
    noise_scale: float = 1.0,
    initial_latent: torch.Tensor | None = None,
) -> LatentState:
    """Create a noised latent state from empty state, conditionings, and noiser.
    Creates an empty latent state, applies conditionings, and then adds noise
    using the provided noiser. Returns the final noised state ready for diffusion.
    """
    state = tools.create_initial_state(device, dtype, initial_latent)
    state = state_with_conditionings(state, conditionings, tools)
    state = noiser(state, noise_scale)

    return state


def state_with_conditionings(
    latent_state: LatentState, conditioning_items: list[ConditioningItem], latent_tools: LatentTools
) -> LatentState:
    """Apply a list of conditionings to a latent state.
    Iterates through the conditioning items and applies each one to the latent
    state in sequence. Returns the modified state with all conditionings applied.
    """
    for conditioning in conditioning_items:
        latent_state = conditioning.apply_to(latent_state=latent_state, latent_tools=latent_tools)

    return latent_state


def post_process_latent(denoised: torch.Tensor, denoise_mask: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Blend denoised output with clean state based on mask."""
    return (denoised * denoise_mask + clean.float() * (1 - denoise_mask)).to(denoised.dtype)


def modality_from_latent_state(
    state: LatentState, context: torch.Tensor, sigma: float | torch.Tensor, enabled: bool = True
) -> Modality:
    """Create a Modality from a latent state.
    Constructs a Modality object with the latent state's data, timesteps derived
    from the denoise mask and sigma, positions, and the provided context.
    """
    return Modality(
        enabled=enabled,
        latent=state.latent,
        timesteps=timesteps_from_mask(state.denoise_mask, sigma),
        positions=state.positions,
        context=context,
        context_mask=None,
    )


def timesteps_from_mask(denoise_mask: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
    """Compute timesteps from a denoise mask and sigma value.
    Multiplies the denoise mask by sigma to produce timesteps for each position
    in the latent state. Areas where the mask is 0 will have zero timesteps.
    """
    return denoise_mask * sigma


def simple_denoising_func(
    video_context: torch.Tensor, audio_context: torch.Tensor, transformer: X0Model
) -> DenoisingFunc:
    def simple_denoising_step(
        video_state: LatentState, audio_state: LatentState, sigmas: torch.Tensor, step_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = sigmas[step_index]
        pos_video = modality_from_latent_state(video_state, video_context, sigma)
        pos_audio = modality_from_latent_state(audio_state, audio_context, sigma)

        denoised_video, denoised_audio = transformer(video=pos_video, audio=pos_audio, perturbations=None)
        return denoised_video, denoised_audio

    return simple_denoising_step


def guider_denoising_func(
    guider: GuiderProtocol,
    v_context_p: torch.Tensor,
    v_context_n: torch.Tensor,
    a_context_p: torch.Tensor,
    a_context_n: torch.Tensor,
    transformer: X0Model,
) -> DenoisingFunc:
    def guider_denoising_step(
        video_state: LatentState, audio_state: LatentState, sigmas: torch.Tensor, step_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = sigmas[step_index]
        pos_video = modality_from_latent_state(video_state, v_context_p, sigma)
        pos_audio = modality_from_latent_state(audio_state, a_context_p, sigma)

        denoised_video, denoised_audio = transformer(video=pos_video, audio=pos_audio, perturbations=None)
        if guider.enabled():
            neg_video = modality_from_latent_state(video_state, v_context_n, sigma)
            neg_audio = modality_from_latent_state(audio_state, a_context_n, sigma)

            neg_denoised_video, neg_denoised_audio = transformer(video=neg_video, audio=neg_audio, perturbations=None)

            denoised_video = denoised_video + guider.delta(denoised_video, neg_denoised_video)
            denoised_audio = denoised_audio + guider.delta(denoised_audio, neg_denoised_audio)

        return denoised_video, denoised_audio

    return guider_denoising_step


def denoise_audio_video(  # noqa: PLR0913
    output_shape: VideoPixelShape,
    conditionings: list[ConditioningItem],
    noiser: Noiser,
    sigmas: torch.Tensor,
    stepper: DiffusionStepProtocol,
    denoising_loop_fn: DenoisingLoopFunc,
    components: PipelineComponents,
    dtype: torch.dtype,
    device: torch.device,
    noise_scale: float = 1.0,
    initial_video_latent: torch.Tensor | None = None,
    initial_audio_latent: torch.Tensor | None = None,
    audio_conditionings: list[ConditioningItem] | None = None,
) -> tuple[LatentState, LatentState]:
    video_state, video_tools = noise_video_state(
        output_shape=output_shape,
        noiser=noiser,
        conditionings=conditionings,
        components=components,
        dtype=dtype,
        device=device,
        noise_scale=noise_scale,
        initial_latent=initial_video_latent,
    )
    audio_state, audio_tools = noise_audio_state(
        output_shape=output_shape,
        noiser=noiser,
        conditionings=audio_conditionings if audio_conditionings is not None else [],
        components=components,
        dtype=dtype,
        device=device,
        noise_scale=noise_scale,
        initial_latent=initial_audio_latent,
    )

    video_state, audio_state = denoising_loop_fn(
        sigmas,
        video_state,
        audio_state,
        stepper,
    )

    video_state = video_tools.clear_conditioning(video_state)
    video_state = video_tools.unpatchify(video_state)
    audio_state = audio_tools.clear_conditioning(audio_state)
    audio_state = audio_tools.unpatchify(audio_state)

    return video_state, audio_state


_UNICODE_REPLACEMENTS = str.maketrans("\u2018\u2019\u201c\u201d\u2014\u2013\u00a0\u2032\u2212", "''\"\"-- '-")


def clean_response(text: str) -> str:
    """Clean a response from curly quotes and leading non-letter characters which Gemma tends to insert."""
    text = text.translate(_UNICODE_REPLACEMENTS)

    # Remove leading non-letter characters
    for i, char in enumerate(text):
        if char.isalpha():
            return text[i:]
    return text


def generate_enhanced_prompt(
    text_encoder: GemmaTextEncoderModelBase,
    prompt: str,
    image_path: str | None = None,
    image_long_side: int = 896,
    seed: int = 42,
) -> str:
    """Generate an enhanced prompt from a text encoder and a prompt."""
    image = None
    if image_path:
        image = decode_image(image_path=image_path)
        image = torch.tensor(image)
        image = resize_aspect_ratio_preserving(image, image_long_side).to(torch.uint8)
        prompt = text_encoder.enhance_i2v(prompt, image, seed=seed)
    else:
        prompt = text_encoder.enhance_t2v(prompt, seed=seed)
    logging.info(f"Enhanced prompt: {prompt}")
    return clean_response(prompt)


def assert_resolution(height: int, width: int, is_two_stage: bool) -> None:
    """Assert that the resolution is divisible by 32."""
    if height % 32 != 0 or width % 32 != 0:
        raise ValueError(
            f"Resolution ({height}x{width}) is not divisible by 32. "
            f"Height and width must be multiples of 32."
        )


# =============================================================================
# Latent Normalization Functions
# =============================================================================
# Ported from ComfyUI-LTXVideo/latent_norm.py
# These functions help fix overbaking and audio clipping issues by normalizing
# latent values during the denoising process.


def adain_normalize(
    latent: torch.Tensor,
    reference: torch.Tensor,
    factor: float = 1.0,
    per_frame: bool = False,
) -> torch.Tensor:
    """
    Adaptive Instance Normalization (AdaIN) for latents.

    Normalizes the input latent to match the mean and standard deviation
    of a reference latent. This helps prevent overbaking by keeping latent
    statistics within expected bounds.

    Args:
        latent: Input latent tensor [B, C, F, H, W] or [B, C, T] for audio
        reference: Reference latent tensor with target statistics
        factor: Blending factor (0=original, 1=fully normalized)
        per_frame: If True, normalize each frame independently

    Returns:
        Normalized latent tensor
    """
    if factor == 0.0:
        return latent

    t = latent.clone()
    ndim = t.ndim

    # Handle both video [B, C, F, H, W] and audio [B, C, T] latents
    if ndim == 5:  # Video: B x C x F x H x W
        if per_frame:
            if reference.size(2) == 1:
                # Broadcast single reference frame to all frames
                reference = reference.repeat(1, 1, t.size(2), 1, 1)
            elif t.size(2) > reference.size(2):
                raise ValueError("Latent has more frames than reference")

        for i in range(t.size(0)):  # batch
            for c in range(t.size(1)):  # channel
                if not per_frame:
                    r_sd, r_mean = torch.std_mean(reference[i, c], dim=None)
                    i_sd, i_mean = torch.std_mean(t[i, c], dim=None)
                    if i_sd > 1e-8:
                        t[i, c] = ((t[i, c] - i_mean) / i_sd) * r_sd + r_mean
                else:
                    for f in range(t.size(2)):  # frame
                        r_sd, r_mean = torch.std_mean(reference[i, c, f], dim=None)
                        i_sd, i_mean = torch.std_mean(t[i, c, f], dim=None)
                        if i_sd > 1e-8:
                            t[i, c, f] = ((t[i, c, f] - i_mean) / i_sd) * r_sd + r_mean

    elif ndim == 3:  # Audio: B x C x T
        for i in range(t.size(0)):  # batch
            for c in range(t.size(1)):  # channel
                r_sd, r_mean = torch.std_mean(reference[i, c], dim=None)
                i_sd, i_mean = torch.std_mean(t[i, c], dim=None)
                if i_sd > 1e-8:
                    t[i, c] = ((t[i, c] - i_mean) / i_sd) * r_sd + r_mean

    return torch.lerp(latent, t, factor)


def statistical_normalize(
    latent: torch.Tensor,
    target_mean: float = 0.0,
    target_std: float = 1.0,
    percentile: float = 95.0,
    factor: float = 1.0,
    clip_outliers: bool = False,
) -> torch.Tensor:
    """
    Statistical normalization with percentile-based filtering.

    Normalizes latents to target mean and std values, using percentile
    filtering to exclude outliers when calculating statistics. This helps
    prevent audio clipping and video overbaking.

    Args:
        latent: Input latent tensor [B, C, F, H, W] or [B, C, T]
        target_mean: Target mean value (default: 0.0)
        target_std: Target standard deviation (default: 1.0)
        percentile: Percentile range for statistics (default: 95.0)
        factor: Blending factor (0=original, 1=fully normalized)
        clip_outliers: If True, clip values outside percentile bounds

    Returns:
        Normalized latent tensor
    """
    if factor == 0.0:
        return latent

    t = latent.clone()
    ndim = t.ndim

    # For 95% of distribution, exclude 2.5% from each tail
    lower_percentile = (100 - percentile) / 2
    upper_percentile = 100 - lower_percentile

    if ndim == 5:  # Video: B x C x F x H x W
        for i in range(t.size(0)):  # batch
            for c in range(t.size(1)):  # channel
                channel_data = t[i, c]
                original_shape = channel_data.shape
                channel_flat = channel_data.flatten()

                # Calculate percentiles
                lower_bound = torch.quantile(channel_flat, lower_percentile / 100)
                upper_bound = torch.quantile(channel_flat, upper_percentile / 100)

                # Create mask for values within percentile range
                mask = (channel_flat >= lower_bound) & (channel_flat <= upper_bound)

                if mask.sum() > 0:
                    filtered_data = channel_flat[mask]
                    current_mean = filtered_data.mean()
                    current_std = filtered_data.std()

                    if current_std > 1e-8:
                        # Normalize all values
                        normalized_flat = (
                            (channel_flat - current_mean) / current_std
                        ) * target_std + target_mean

                        if clip_outliers:
                            # Calculate normalized bounds
                            normalized_lower = (
                                (lower_bound - current_mean) / current_std
                            ) * target_std + target_mean
                            normalized_upper = (
                                (upper_bound - current_mean) / current_std
                            ) * target_std + target_mean

                            # Clip outliers
                            normalized_flat = torch.where(
                                channel_flat < lower_bound,
                                normalized_lower,
                                normalized_flat,
                            )
                            normalized_flat = torch.where(
                                channel_flat > upper_bound,
                                normalized_upper,
                                normalized_flat,
                            )

                        t[i, c] = normalized_flat.reshape(original_shape)
                    else:
                        t[i, c] = channel_data - current_mean + target_mean

    elif ndim == 3:  # Audio: B x C x T
        for i in range(t.size(0)):  # batch
            for c in range(t.size(1)):  # channel
                channel_data = t[i, c]
                channel_flat = channel_data.flatten()

                lower_bound = torch.quantile(channel_flat, lower_percentile / 100)
                upper_bound = torch.quantile(channel_flat, upper_percentile / 100)

                mask = (channel_flat >= lower_bound) & (channel_flat <= upper_bound)

                if mask.sum() > 0:
                    filtered_data = channel_flat[mask]
                    current_mean = filtered_data.mean()
                    current_std = filtered_data.std()

                    if current_std > 1e-8:
                        normalized_flat = (
                            (channel_flat - current_mean) / current_std
                        ) * target_std + target_mean

                        if clip_outliers:
                            normalized_lower = (
                                (lower_bound - current_mean) / current_std
                            ) * target_std + target_mean
                            normalized_upper = (
                                (upper_bound - current_mean) / current_std
                            ) * target_std + target_mean
                            normalized_flat = torch.clamp(
                                normalized_flat, normalized_lower, normalized_upper
                            )

                        t[i, c] = normalized_flat
                    else:
                        t[i, c] = channel_data - current_mean + target_mean

    return torch.lerp(latent, t, factor)


def create_per_step_stat_norm_fn(
    factors: str | list[float],
    target_mean: float = 0.0,
    target_std: float = 1.0,
    percentile: float = 95.0,
    clip_outliers: bool = False,
    apply_to_video: bool = True,
    apply_to_audio: bool = True,
):
    """
    Create a per-step statistical normalization function for use in the denoising loop.

    This function returns a callable that normalizes latents after each denoising
    step, with different strengths at different steps. This is the recommended
    approach for fixing overbaking/clipping - apply stronger normalization early
    in the denoising process and reduce it later.

    Args:
        factors: Comma-separated string or list of factors for each step.
                 e.g., "0.9,0.75,0.5,0.25,0.0" applies 0.9 at step 0, etc.
                 The last factor is used for all remaining steps.
        target_mean: Target mean for normalization (default: 0.0)
        target_std: Target std for normalization (default: 1.0)
        percentile: Percentile for outlier filtering (default: 95.0)
        clip_outliers: Whether to clip outliers (default: False)
        apply_to_video: Apply normalization to video latents (default: True)
        apply_to_audio: Apply normalization to audio latents (default: True)

    Returns:
        A function with signature:
        fn(video_latent, audio_latent, step_idx) -> (video_latent, audio_latent)
    """
    if isinstance(factors, str):
        factor_list = [float(x.strip()) for x in factors.split(",")]
    else:
        factor_list = list(factors)

    def norm_fn(
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        step_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        factor = factor_list[min(step_idx, len(factor_list) - 1)]

        if factor == 0.0:
            return video_latent, audio_latent

        if apply_to_video:
            video_latent = statistical_normalize(
                video_latent,
                target_mean=target_mean,
                target_std=target_std,
                percentile=percentile,
                factor=factor,
                clip_outliers=clip_outliers,
            )

        if apply_to_audio:
            audio_latent = statistical_normalize(
                audio_latent,
                target_mean=target_mean,
                target_std=target_std,
                percentile=percentile,
                factor=factor,
                clip_outliers=clip_outliers,
            )

        return video_latent, audio_latent

    return norm_fn


def create_per_step_adain_norm_fn(
    factors: str | list[float],
    reference_video: torch.Tensor | None = None,
    reference_audio: torch.Tensor | None = None,
    per_frame: bool = False,
):
    """
    Create a per-step AdaIN normalization function for use in the denoising loop.

    This uses a reference latent to guide the statistics of the generated latent.
    Useful when you have a "known good" latent to match against.

    Args:
        factors: Comma-separated string or list of factors for each step.
        reference_video: Reference video latent for AdaIN (optional)
        reference_audio: Reference audio latent for AdaIN (optional)
        per_frame: Apply AdaIN per-frame for video (default: False)

    Returns:
        A function with signature:
        fn(video_latent, audio_latent, step_idx) -> (video_latent, audio_latent)
    """
    if isinstance(factors, str):
        factor_list = [float(x.strip()) for x in factors.split(",")]
    else:
        factor_list = list(factors)

    def norm_fn(
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        step_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        factor = factor_list[min(step_idx, len(factor_list) - 1)]

        if factor == 0.0:
            return video_latent, audio_latent

        if reference_video is not None:
            video_latent = adain_normalize(
                video_latent,
                reference_video,
                factor=factor,
                per_frame=per_frame,
            )

        if reference_audio is not None:
            audio_latent = adain_normalize(
                audio_latent,
                reference_audio,
                factor=factor,
                per_frame=False,
            )

        return video_latent, audio_latent

    return norm_fn
