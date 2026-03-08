import torch

from ltx_core.conditioning.item import ConditioningItem
from ltx_core.tools import LatentTools
from ltx_core.types import LatentState


class AudioConditionByLatent(ConditioningItem):
    def __init__(self, latent: torch.Tensor, strength: float = 1.0):
        self.latent = latent
        self.strength = strength

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        tokens = latent_tools.patchifier.patchify(self.latent)

        latent_state = latent_state.clone()

        num_tokens = min(tokens.shape[1], latent_state.latent.shape[1])
        latent_state.latent[:, :num_tokens] = tokens[:, :num_tokens]
        latent_state.clean_latent[:, :num_tokens] = tokens[:, :num_tokens]
        latent_state.denoise_mask[:, :num_tokens] = 1.0 - self.strength

        return latent_state
