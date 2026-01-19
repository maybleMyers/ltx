import torch

from ltx_core.components.patchifiers import get_pixel_coords
from ltx_core.conditioning.item import ConditioningItem
from ltx_core.tools import VideoLatentTools
from ltx_core.types import LatentState, VideoLatentShape


class VideoConditionByGuideLatent(ConditioningItem):
    """
    Conditions video by appending guide latents as surplus tokens.
    These tokens influence attention but are cropped by clear_conditioning() after denoising.
    Unlike VideoConditionByLatentIndex (which replaces), this appends for CropGuides pattern.
    """

    def __init__(self, latent: torch.Tensor, frame_idx: int, strength: float = 0.98):
        """
        Args:
            latent: Guide latent tensor [B, C, T, H, W] - already encoded
            frame_idx: Pixel frame index for RoPE position encoding (can be negative)
            strength: How much to preserve guide (0.98 = 2% denoising, 98% preserved)
        """
        self.latent = latent
        self.frame_idx = frame_idx
        self.strength = strength

    def apply_to(self, latent_state: LatentState, latent_tools: VideoLatentTools) -> LatentState:
        target_device = latent_state.latent.device
        guide = self.latent.to(target_device)

        tokens = latent_tools.patchifier.patchify(guide)
        latent_coords = latent_tools.patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape.from_torch_shape(guide.shape),
            device=target_device,
        )
        positions = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=latent_tools.scale_factors,
            causal_fix=False,  # No causal fix for guides
        )

        positions[:, 0, ...] += self.frame_idx
        positions = positions.to(dtype=torch.float32, device=target_device)
        positions[:, 0, ...] /= latent_tools.fps

        denoise_mask = torch.full(
            size=(*tokens.shape[:2], 1),
            fill_value=1.0 - self.strength,  # 0.98 strength -> 0.02 denoise_mask
            device=target_device,
            dtype=guide.dtype,
        )

        # APPEND tokens (CropGuides pattern - these get cropped by clear_conditioning)
        return LatentState(
            latent=torch.cat([latent_state.latent, tokens], dim=1),
            denoise_mask=torch.cat([latent_state.denoise_mask, denoise_mask], dim=1),
            positions=torch.cat([latent_state.positions, positions], dim=2),
            clean_latent=torch.cat([latent_state.clean_latent, tokens], dim=1),
        )
