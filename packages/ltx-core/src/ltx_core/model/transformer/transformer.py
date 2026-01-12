from dataclasses import dataclass, replace

import torch

from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from ltx_core.model.transformer.attention import Attention, AttentionCallable, AttentionFunction
from ltx_core.model.transformer.feed_forward import FeedForward
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.transformer_args import TransformerArgs
from ltx_core.utils import rms_norm


# Helper functions for positional embeddings (which are tuples of (cos, sin) tensors)
def _pe_to_cpu(pe: tuple[torch.Tensor, torch.Tensor] | None) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Move positional embedding tuple to CPU."""
    if pe is None:
        return None
    return (pe[0].cpu(), pe[1].cpu())


def _pe_to_device(pe: tuple[torch.Tensor, torch.Tensor] | None, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Move positional embedding tuple to device."""
    if pe is None:
        return None
    return (pe[0].to(device, non_blocking=True), pe[1].to(device, non_blocking=True))


def _pe_slice(pe: tuple[torch.Tensor, torch.Tensor] | None, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Slice positional embedding tuple along token dimension.

    For interleaved rope (3D tensors: B, T, D), slices dim 1.
    For split rope (4D tensors: B, H, T, D), slices dim 2.
    """
    if pe is None:
        return None
    # Auto-detect dimension based on tensor shape
    if pe[0].ndim == 3:
        # Interleaved rope: (B, T, D) - slice dim 1
        return (pe[0][:, start:end], pe[1][:, start:end])
    elif pe[0].ndim == 4:
        # Split rope: (B, H, T, D) - slice dim 2
        return (pe[0][:, :, start:end], pe[1][:, :, start:end])
    else:
        raise ValueError(f"Unsupported positional embedding shape: {pe[0].shape}")


@dataclass
class TransformerConfig:
    dim: int
    heads: int
    d_head: int
    context_dim: int


class BasicAVTransformerBlock(torch.nn.Module):
    # FFN chunk size for memory optimization (None = disabled, set during inference for long sequences)
    ffn_chunk_size: int | None = None
    # Temporal chunk size for processing long videos (None = disabled)
    temporal_chunk_size: int | None = None

    def __init__(
        self,
        idx: int,
        video: TransformerConfig | None = None,
        audio: TransformerConfig | None = None,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        norm_eps: float = 1e-6,
        attention_function: AttentionFunction | AttentionCallable = AttentionFunction.DEFAULT,
    ):
        super().__init__()

        self.idx = idx
        if video is not None:
            self.attn1 = Attention(
                query_dim=video.dim,
                heads=video.heads,
                dim_head=video.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
            )
            self.attn2 = Attention(
                query_dim=video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
            )
            self.ff = FeedForward(video.dim, dim_out=video.dim)
            self.scale_shift_table = torch.nn.Parameter(torch.empty(6, video.dim))

        if audio is not None:
            self.audio_attn1 = Attention(
                query_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
            )
            self.audio_attn2 = Attention(
                query_dim=audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
            )
            self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim)
            self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(6, audio.dim))

        if audio is not None and video is not None:
            # Q: Video, K,V: Audio
            self.audio_to_video_attn = Attention(
                query_dim=video.dim,
                context_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
            )

            # Q: Audio, K,V: Video
            self.video_to_audio_attn = Attention(
                query_dim=audio.dim,
                context_dim=video.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
            )

            self.scale_shift_table_a2v_ca_audio = torch.nn.Parameter(torch.empty(5, audio.dim))
            self.scale_shift_table_a2v_ca_video = torch.nn.Parameter(torch.empty(5, video.dim))

        self.norm_eps = norm_eps

    def get_ada_values(
        self, scale_shift_table: torch.Tensor, batch_size: int, timestep: torch.Tensor, indices: slice
    ) -> tuple[torch.Tensor, ...]:
        num_ada_params = scale_shift_table.shape[0]

        ada_values = (
            scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        ).unbind(dim=2)
        return ada_values

    def get_av_ca_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        scale_shift_timestep: torch.Tensor,
        gate_timestep: torch.Tensor,
        num_scale_shift_values: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_shift_ada_values = self.get_ada_values(
            scale_shift_table[:num_scale_shift_values, :], batch_size, scale_shift_timestep, slice(None, None)
        )
        gate_ada_values = self.get_ada_values(
            scale_shift_table[num_scale_shift_values:, :], batch_size, gate_timestep, slice(None, None)
        )

        scale_shift_chunks = [t.squeeze(2) for t in scale_shift_ada_values]
        gate_ada_values = [t.squeeze(2) for t in gate_ada_values]

        return (*scale_shift_chunks, *gate_ada_values)

    def forward(  # noqa: PLR0915
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig | None = None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        batch_size = video.x.shape[0]
        if perturbations is None:
            perturbations = BatchedPerturbationConfig.empty(batch_size)

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        run_vx = video is not None and video.enabled and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax.numel() > 0

        run_a2v = run_vx and (audio is not None and audio.enabled and ax.numel() > 0)
        run_v2a = run_ax and (video is not None and video.enabled and vx.numel() > 0)

        if run_vx:
            vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3)
            )
            if not perturbations.all_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx):
                norm_vx = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_msa) + vshift_msa
                v_mask = perturbations.mask_like(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx, vx)
                vx = vx + self.attn1(norm_vx, pe=video.positional_embeddings) * vgate_msa * v_mask

            vx = vx + self.attn2(rms_norm(vx, eps=self.norm_eps), context=video.context, mask=video.context_mask)

            del vshift_msa, vscale_msa, vgate_msa

        if run_ax:
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3)
            )

            if not perturbations.all_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx):
                norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_msa) + ashift_msa
                a_mask = perturbations.mask_like(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx, ax)
                ax = ax + self.audio_attn1(norm_ax, pe=audio.positional_embeddings) * agate_msa * a_mask

            ax = ax + self.audio_attn2(rms_norm(ax, eps=self.norm_eps), context=audio.context, mask=audio.context_mask)

            del ashift_msa, ascale_msa, agate_msa

        # Audio - Video cross attention.
        if run_a2v or run_v2a:
            vx_norm3 = rms_norm(vx, eps=self.norm_eps)
            ax_norm3 = rms_norm(ax, eps=self.norm_eps)

            (
                scale_ca_audio_hidden_states_a2v,
                shift_ca_audio_hidden_states_a2v,
                scale_ca_audio_hidden_states_v2a,
                shift_ca_audio_hidden_states_v2a,
                gate_out_v2a,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_audio,
                ax.shape[0],
                audio.cross_scale_shift_timestep,
                audio.cross_gate_timestep,
            )

            (
                scale_ca_video_hidden_states_a2v,
                shift_ca_video_hidden_states_a2v,
                scale_ca_video_hidden_states_v2a,
                shift_ca_video_hidden_states_v2a,
                gate_out_a2v,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_video,
                vx.shape[0],
                video.cross_scale_shift_timestep,
                video.cross_gate_timestep,
            )

            if run_a2v:
                vx_scaled = vx_norm3 * (1 + scale_ca_video_hidden_states_a2v) + shift_ca_video_hidden_states_a2v
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_hidden_states_a2v) + shift_ca_audio_hidden_states_a2v
                a2v_mask = perturbations.mask_like(PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx, vx)
                vx = vx + (
                    self.audio_to_video_attn(
                        vx_scaled,
                        context=ax_scaled,
                        pe=video.cross_positional_embeddings,
                        k_pe=audio.cross_positional_embeddings,
                    )
                    * gate_out_a2v
                    * a2v_mask
                )

            if run_v2a:
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_hidden_states_v2a) + shift_ca_audio_hidden_states_v2a
                vx_scaled = vx_norm3 * (1 + scale_ca_video_hidden_states_v2a) + shift_ca_video_hidden_states_v2a
                v2a_mask = perturbations.mask_like(PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx, ax)
                ax = ax + (
                    self.video_to_audio_attn(
                        ax_scaled,
                        context=vx_scaled,
                        pe=audio.cross_positional_embeddings,
                        k_pe=video.cross_positional_embeddings,
                    )
                    * gate_out_v2a
                    * v2a_mask
                )

            del gate_out_a2v, gate_out_v2a
            del (
                scale_ca_video_hidden_states_a2v,
                shift_ca_video_hidden_states_a2v,
                scale_ca_audio_hidden_states_a2v,
                shift_ca_audio_hidden_states_a2v,
                scale_ca_video_hidden_states_v2a,
                shift_ca_video_hidden_states_v2a,
                scale_ca_audio_hidden_states_v2a,
                shift_ca_audio_hidden_states_v2a,
            )

        if run_vx:
            vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, None)
            )
            vx_scaled = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_mlp) + vshift_mlp
            # Use chunked FFN for long sequences to reduce peak memory (only during inference)
            if not self.training and self.ffn_chunk_size is not None:
                vx = vx + self.ff.forward_chunked(vx_scaled, self.ffn_chunk_size) * vgate_mlp
            else:
                vx = vx + self.ff(vx_scaled) * vgate_mlp

            del vshift_mlp, vscale_mlp, vgate_mlp

        if run_ax:
            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, None)
            )
            ax_scaled = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_mlp) + ashift_mlp
            # Use chunked FFN for long sequences to reduce peak memory (only during inference)
            if not self.training and self.ffn_chunk_size is not None:
                ax = ax + self.audio_ff.forward_chunked(ax_scaled, self.ffn_chunk_size) * agate_mlp
            else:
                ax = ax + self.audio_ff(ax_scaled) * agate_mlp

            del ashift_mlp, ascale_mlp, agate_mlp

        return replace(video, x=vx) if video is not None else None, replace(audio, x=ax) if audio is not None else None

    def forward_chunked(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig | None = None,
        chunk_size: int = 400000,
        device: torch.device | str = "cuda",
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        """
        Chunked forward pass for very long videos that don't fit in GPU memory.

        Video tensors (vx) are kept on CPU and processed in chunks. Audio is small
        enough to fit on GPU and is processed normally. This maintains full attention
        context via streaming K/V from CPU.

        Args:
            video: TransformerArgs with video.x on CPU
            audio: TransformerArgs with audio.x (can be on CPU or GPU)
            perturbations: Optional perturbation config
            chunk_size: Number of video tokens per chunk
            device: GPU device for computation

        Returns:
            Updated (video, audio) TransformerArgs with x tensors on CPU
        """
        device = torch.device(device) if isinstance(device, str) else device
        batch_size = video.x.shape[0] if video is not None else audio.x.shape[0]

        if perturbations is None:
            perturbations = BatchedPerturbationConfig.empty(batch_size)

        # Get tensors - video on CPU, audio moves to GPU (it's small)
        vx = video.x if video is not None else None  # On CPU
        ax = audio.x.to(device) if audio is not None else None  # Move to GPU

        run_vx = video is not None and video.enabled and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax is not None and ax.numel() > 0

        run_a2v = run_vx and run_ax
        run_v2a = run_ax and run_vx

        N_video = vx.shape[1] if vx is not None else 0

        # ============================================================
        # 1. VIDEO SELF-ATTENTION (chunked)
        # ============================================================
        if run_vx:
            # Get ada values (small, can stay on GPU)
            vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps.to(device), slice(0, 3)
            )

            if not perturbations.all_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx):
                # Apply norm + ada to input in chunks, store on CPU
                norm_vx_chunks = []
                for start in range(0, N_video, chunk_size):
                    end = min(start + chunk_size, N_video)
                    vx_chunk = vx[:, start:end].to(device)
                    norm_chunk = rms_norm(vx_chunk, eps=self.norm_eps) * (1 + vscale_msa) + vshift_msa
                    norm_vx_chunks.append(norm_chunk.cpu())
                    del vx_chunk, norm_chunk
                norm_vx = torch.cat(norm_vx_chunks, dim=1)
                del norm_vx_chunks

                # Chunked self-attention (streaming K/V from CPU)
                pe = _pe_to_cpu(video.positional_embeddings)
                attn_out = self.attn1.forward_chunked(norm_vx, pe, chunk_size, device)
                del norm_vx, pe

                # Apply gate and residual in chunks
                v_mask = perturbations.mask_like(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx, vx[:, :1].to(device)).cpu()
                vx_new_chunks = []
                for start in range(0, N_video, chunk_size):
                    end = min(start + chunk_size, N_video)
                    vx_chunk = vx[:, start:end].to(device)
                    attn_chunk = attn_out[:, start:end].to(device)
                    vx_chunk = vx_chunk + attn_chunk * vgate_msa * v_mask
                    vx_new_chunks.append(vx_chunk.cpu())
                    del vx_chunk, attn_chunk
                vx = torch.cat(vx_new_chunks, dim=1)
                del vx_new_chunks, attn_out, v_mask

            # Cross-attention to text (text context is small, just chunk queries)
            text_attn_out_chunks = []
            for start in range(0, N_video, chunk_size):
                end = min(start + chunk_size, N_video)
                vx_chunk = vx[:, start:end].to(device)
                norm_chunk = rms_norm(vx_chunk, eps=self.norm_eps)
                attn_chunk = self.attn2(
                    norm_chunk,
                    context=video.context.to(device),
                    mask=video.context_mask.to(device) if video.context_mask is not None else None
                )
                text_attn_out_chunks.append(attn_chunk.cpu())
                del vx_chunk, norm_chunk, attn_chunk
            text_attn_out = torch.cat(text_attn_out_chunks, dim=1)
            del text_attn_out_chunks

            # Apply residual
            vx = vx + text_attn_out
            del text_attn_out, vshift_msa, vscale_msa, vgate_msa
            torch.cuda.empty_cache()

        # ============================================================
        # 2. AUDIO SELF-ATTENTION (non-chunked, audio is small)
        # ============================================================
        if run_ax:
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps.to(device), slice(0, 3)
            )

            if not perturbations.all_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx):
                norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_msa) + ashift_msa
                a_mask = perturbations.mask_like(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx, ax)
                ax = ax + self.audio_attn1(
                    norm_ax,
                    pe=_pe_to_device(audio.positional_embeddings, device)
                ) * agate_msa * a_mask

            ax = ax + self.audio_attn2(
                rms_norm(ax, eps=self.norm_eps),
                context=audio.context.to(device),
                mask=audio.context_mask.to(device) if audio.context_mask is not None else None
            )
            del ashift_msa, ascale_msa, agate_msa

        # ============================================================
        # 3. AUDIO-VIDEO CROSS-ATTENTION (chunk video queries)
        # ============================================================
        if run_a2v or run_v2a:
            # Compute ada values
            (
                scale_ca_audio_a2v, shift_ca_audio_a2v,
                scale_ca_audio_v2a, shift_ca_audio_v2a,
                gate_out_v2a,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_audio, ax.shape[0],
                audio.cross_scale_shift_timestep.to(device),
                audio.cross_gate_timestep.to(device),
            )

            (
                scale_ca_video_a2v, shift_ca_video_a2v,
                scale_ca_video_v2a, shift_ca_video_v2a,
                gate_out_a2v,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_video, vx.shape[0],
                video.cross_scale_shift_timestep.to(device),
                video.cross_gate_timestep.to(device),
            )

            # Prepare normalized audio (on GPU, reusable)
            ax_norm3 = rms_norm(ax, eps=self.norm_eps)
            ax_scaled_a2v = ax_norm3 * (1 + scale_ca_audio_a2v) + shift_ca_audio_a2v
            ax_scaled_v2a = ax_norm3 * (1 + scale_ca_audio_v2a) + shift_ca_audio_v2a

            if run_a2v:
                # Audio-to-video: video queries, audio K/V (audio is small, fits on GPU)
                a2v_out_chunks = []
                a2v_mask = perturbations.mask_like(PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx, vx[:, :1].to(device)).cpu()

                for start in range(0, N_video, chunk_size):
                    end = min(start + chunk_size, N_video)
                    vx_chunk = vx[:, start:end].to(device)
                    vx_norm_chunk = rms_norm(vx_chunk, eps=self.norm_eps)
                    vx_scaled_chunk = vx_norm_chunk * (1 + scale_ca_video_a2v) + shift_ca_video_a2v

                    # Cross-attention: video queries attend to audio
                    v_pe_chunk = _pe_to_device(_pe_slice(video.cross_positional_embeddings, start, end), device)
                    a2v_chunk = self.audio_to_video_attn(
                        vx_scaled_chunk,
                        context=ax_scaled_a2v,
                        pe=v_pe_chunk,
                        k_pe=_pe_to_device(audio.cross_positional_embeddings, device),
                    )
                    a2v_out_chunks.append((vx_chunk + a2v_chunk * gate_out_a2v * a2v_mask).cpu())
                    del vx_chunk, vx_norm_chunk, vx_scaled_chunk, a2v_chunk, v_pe_chunk

                vx = torch.cat(a2v_out_chunks, dim=1)
                del a2v_out_chunks, a2v_mask

            if run_v2a:
                # Video-to-audio: audio queries, video K/V (need to stream video)
                # Prepare video K/V by computing in chunks and storing on CPU
                V_kv_chunks = []
                for start in range(0, N_video, chunk_size):
                    end = min(start + chunk_size, N_video)
                    vx_chunk = vx[:, start:end].to(device)
                    vx_norm_chunk = rms_norm(vx_chunk, eps=self.norm_eps)
                    vx_scaled_chunk = vx_norm_chunk * (1 + scale_ca_video_v2a) + shift_ca_video_v2a
                    V_kv_chunks.append(vx_scaled_chunk.cpu())
                    del vx_chunk, vx_norm_chunk, vx_scaled_chunk

                vx_scaled_full = torch.cat(V_kv_chunks, dim=1)
                del V_kv_chunks

                # Use chunked cross-attention with video as context
                v2a_out = self.video_to_audio_attn.forward_chunked_cross(
                    x=ax_scaled_v2a.cpu(),
                    context=vx_scaled_full,
                    pe=_pe_to_cpu(audio.cross_positional_embeddings),
                    k_pe=_pe_to_cpu(video.cross_positional_embeddings),
                    chunk_size=chunk_size,
                    device=device,
                ).to(device)

                v2a_mask = perturbations.mask_like(PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx, ax)
                ax = ax + v2a_out * gate_out_v2a * v2a_mask
                del vx_scaled_full, v2a_out, v2a_mask

            del ax_norm3, ax_scaled_a2v, ax_scaled_v2a
            del scale_ca_audio_a2v, shift_ca_audio_a2v, scale_ca_audio_v2a, shift_ca_audio_v2a
            del scale_ca_video_a2v, shift_ca_video_a2v, scale_ca_video_v2a, shift_ca_video_v2a
            del gate_out_a2v, gate_out_v2a
            torch.cuda.empty_cache()

        # ============================================================
        # 4. FFN (chunked for video, normal for audio)
        # ============================================================
        if run_vx:
            vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps.to(device), slice(3, None)
            )

            vx_ffn_chunks = []
            for start in range(0, N_video, chunk_size):
                end = min(start + chunk_size, N_video)
                vx_chunk = vx[:, start:end].to(device)
                vx_scaled = rms_norm(vx_chunk, eps=self.norm_eps) * (1 + vscale_mlp) + vshift_mlp

                if self.ffn_chunk_size is not None:
                    ffn_out = self.ff.forward_chunked(vx_scaled, self.ffn_chunk_size)
                else:
                    ffn_out = self.ff(vx_scaled)

                vx_ffn_chunks.append((vx_chunk + ffn_out * vgate_mlp).cpu())
                del vx_chunk, vx_scaled, ffn_out

            vx = torch.cat(vx_ffn_chunks, dim=1)
            del vx_ffn_chunks, vshift_mlp, vscale_mlp, vgate_mlp

        if run_ax:
            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps.to(device), slice(3, None)
            )
            ax_scaled = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_mlp) + ashift_mlp

            if self.ffn_chunk_size is not None:
                ax = ax + self.audio_ff.forward_chunked(ax_scaled, self.ffn_chunk_size) * agate_mlp
            else:
                ax = ax + self.audio_ff(ax_scaled) * agate_mlp

            del ashift_mlp, ascale_mlp, agate_mlp

        torch.cuda.empty_cache()

        # Return with tensors on CPU (video) and GPU (audio, but we'll move to CPU)
        ax_cpu = ax.cpu() if ax is not None else None
        return (
            replace(video, x=vx) if video is not None else None,
            replace(audio, x=ax_cpu) if audio is not None else None
        )
