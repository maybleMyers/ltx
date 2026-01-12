from enum import Enum
from typing import Protocol

import torch

from ltx_core.model.transformer.rope import LTXRopeType, apply_rotary_emb

memory_efficient_attention = None
flash_attn_interface = None
try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    memory_efficient_attention = None
try:
    # FlashAttention3 and XFormersAttention cannot be used together
    if memory_efficient_attention is None:
        import flash_attn_interface
except ImportError:
    flash_attn_interface = None


# Type alias for positional embeddings (tuple of cos, sin tensors)
PETuple = tuple[torch.Tensor, torch.Tensor] | None


def _pe_to_device(pe: PETuple, device: torch.device | str) -> PETuple:
    """Move positional embedding tuple to device."""
    if pe is None:
        return None
    return (pe[0].to(device, non_blocking=True), pe[1].to(device, non_blocking=True))


def _pe_slice(pe: PETuple, start: int, end: int) -> PETuple:
    """Slice positional embedding tuple along token dimension.

    For interleaved rope (3D tensors: B, T, D), slices dim 1.
    For split rope (4D tensors: B, H, T, D), slices dim 2.
    """
    if pe is None:
        return None
    if pe[0].ndim == 3:
        return (pe[0][:, start:end], pe[1][:, start:end])
    elif pe[0].ndim == 4:
        return (pe[0][:, :, start:end], pe[1][:, :, start:end])
    else:
        raise ValueError(f"Unsupported positional embedding shape: {pe[0].shape}")


class AttentionCallable(Protocol):
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor: ...


class PytorchAttention(AttentionCallable):
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))

        if mask is not None:
            # add a batch dimension if there isn't already one
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # add a heads dimension if there isn't already one
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        return out


class XFormersAttention(AttentionCallable):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory_efficient_attention is None:
            raise RuntimeError("XFormersAttention was selected but `xformers` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        # xformers expects [B, M, H, K]
        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        if mask is not None:
            # add a singleton batch dimension
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # add a singleton heads dimension
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            # pad to a multiple of 8
            pad = 8 - mask.shape[-1] % 8
            # the xformers docs says that it's allowed to have a mask of shape (1, Nq, Nk)
            # but when using separated heads, the shape has to be (B, H, Nq, Nk)
            # in flux, this matrix ends up being over 1GB
            # here, we create a mask with the same batch/head size as the input mask (potentially singleton or full)
            mask_out = torch.empty(
                [mask.shape[0], mask.shape[1], q.shape[1], mask.shape[-1] + pad], dtype=q.dtype, device=q.device
            )

            mask_out[..., : mask.shape[-1]] = mask
            # doesn't this remove the padding again??
            mask = mask_out[..., : mask.shape[-1]]
            mask = mask.expand(b, heads, -1, -1)

        out = memory_efficient_attention(q.to(v.dtype), k.to(v.dtype), v, attn_bias=mask, p=0.0)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class FlashAttention3(AttentionCallable):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if flash_attn_interface is None:
            raise RuntimeError("FlashAttention3 was selected but `FlashAttention3` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        if mask is not None:
            raise NotImplementedError("Mask is not supported for FlashAttention3")

        out = flash_attn_interface.flash_attn_func(q.to(v.dtype), k.to(v.dtype), v)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class AttentionFunction(Enum):
    PYTORCH = "pytorch"
    XFORMERS = "xformers"
    FLASH_ATTENTION_3 = "flash_attention_3"
    DEFAULT = "default"

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self is AttentionFunction.PYTORCH:
            return PytorchAttention()(q, k, v, heads, mask)
        elif self is AttentionFunction.XFORMERS:
            return XFormersAttention()(q, k, v, heads, mask)
        elif self is AttentionFunction.FLASH_ATTENTION_3:
            return FlashAttention3()(q, k, v, heads, mask)
        else:
            # Default behavior: XFormers if installed else - PyTorch
            return (
                XFormersAttention()(q, k, v, heads, mask)
                if memory_efficient_attention is not None
                else PytorchAttention()(q, k, v, heads, mask)
            )


class Attention(torch.nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        attention_function: AttentionCallable | AttentionFunction = AttentionFunction.DEFAULT,
    ) -> None:
        super().__init__()
        self.rope_type = rope_type
        self.attention_function = attention_function

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head

        self.q_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=True)

        self.to_out = torch.nn.Sequential(torch.nn.Linear(inner_dim, query_dim, bias=True), torch.nn.Identity())

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.to_q(x)
        context = x if context is None else context
        k = self.to_k(context)
        v = self.to_v(context)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if pe is not None:
            q = apply_rotary_emb(q, pe, self.rope_type)
            k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)

        # attention_function can be an enum *or* a custom callable
        out = self.attention_function(q, k, v, self.heads, mask)
        return self.to_out(out)

    def forward_chunked(
        self,
        x: torch.Tensor,
        pe: torch.Tensor | None,
        chunk_size: int,
        device: torch.device | str,
        attention_chunk_size: int = 4096,
    ) -> torch.Tensor:
        """
        Chunked self-attention that maintains full global context.

        Computes K/V for all tokens (stored on CPU), then processes Q in chunks,
        streaming K/V from CPU for each Q chunk. Uses online softmax for numerical
        stability without materializing full attention matrix.

        Args:
            x: [B, N, D] input tensor (can be on CPU)
            pe: positional embeddings tuple (cos, sin) (can be on CPU)
            chunk_size: number of tokens per chunk for outer operations
            device: GPU device to use for computation
            attention_chunk_size: smaller chunk size for attention computation (default 4096)

        Returns:
            [B, N, D] attention output (on CPU)
        """
        B, N, D = x.shape
        device = torch.device(device) if isinstance(device, str) else device

        # Use smaller chunks for attention to avoid huge score matrices
        # With 4096 Q and 4096 K: scores = [B, H, 4096, 4096] ≈ 2GB at fp32
        attn_chunk = min(attention_chunk_size, chunk_size)

        # Phase 1: Compute K, V for all tokens in small chunks, store on CPU
        K_chunks = []
        V_chunks = []

        for start in range(0, N, attn_chunk):
            end = min(start + attn_chunk, N)
            x_chunk = x[:, start:end].to(device, non_blocking=True)
            pe_chunk = _pe_to_device(_pe_slice(pe, start, end), device)

            # Compute K, V for this chunk
            k = self.k_norm(self.to_k(x_chunk))
            v = self.to_v(x_chunk)

            if pe_chunk is not None:
                k = apply_rotary_emb(k, pe_chunk, self.rope_type)

            # Store on CPU
            K_chunks.append(k.cpu())
            V_chunks.append(v.cpu())

            del x_chunk, k, v, pe_chunk
            torch.cuda.empty_cache()

        # Phase 2: For each Q chunk, attend to ALL K/V via streaming
        output_chunks = []

        for q_start in range(0, N, attn_chunk):
            q_end = min(q_start + attn_chunk, N)
            x_chunk = x[:, q_start:q_end].to(device, non_blocking=True)
            pe_chunk = _pe_to_device(_pe_slice(pe, q_start, q_end), device)

            # Compute Q for this chunk
            q = self.q_norm(self.to_q(x_chunk))
            if pe_chunk is not None:
                q = apply_rotary_emb(q, pe_chunk, self.rope_type)

            # Attend to ALL K/V by streaming from CPU
            attn_out = self._streamed_attention(q, K_chunks, V_chunks, device)

            # Apply output projection and store on CPU
            out = self.to_out(attn_out)
            output_chunks.append(out.cpu())

            del x_chunk, pe_chunk, q, attn_out, out
            torch.cuda.empty_cache()

        # Clean up K/V chunks
        del K_chunks, V_chunks

        return torch.cat(output_chunks, dim=1)

    def forward_chunked_cross(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        pe: torch.Tensor | None,
        k_pe: torch.Tensor | None,
        chunk_size: int,
        device: torch.device | str,
        mask: torch.Tensor | None = None,
        attention_chunk_size: int = 4096,
    ) -> torch.Tensor:
        """
        Chunked cross-attention where context (K/V source) fits in memory.

        For cross-attention to text embeddings, the context is small enough to
        keep on GPU. We just chunk the queries.

        Args:
            x: [B, N, D] query input (can be on CPU)
            context: [B, M, D] context for K/V (should fit on GPU)
            pe: positional embeddings for Q
            k_pe: positional embeddings for K
            chunk_size: number of query tokens per chunk (outer operations)
            device: GPU device
            mask: optional attention mask
            attention_chunk_size: smaller chunk size for attention computation (default 4096)

        Returns:
            [B, N, D] attention output (on CPU)
        """
        B, N, D = x.shape
        device = torch.device(device) if isinstance(device, str) else device

        # Use smaller chunks for attention to avoid huge score matrices
        attn_chunk = min(attention_chunk_size, chunk_size)

        # Move context to GPU and compute K, V once
        context_gpu = context.to(device, non_blocking=True)
        k = self.k_norm(self.to_k(context_gpu))
        v = self.to_v(context_gpu)

        k_pe_gpu = _pe_to_device(k_pe, device)
        if k_pe_gpu is not None:
            k = apply_rotary_emb(k, k_pe_gpu, self.rope_type)

        # Process Q in small chunks for memory efficiency
        output_chunks = []

        for q_start in range(0, N, attn_chunk):
            q_end = min(q_start + attn_chunk, N)
            x_chunk = x[:, q_start:q_end].to(device, non_blocking=True)
            pe_chunk = _pe_to_device(_pe_slice(pe, q_start, q_end), device)

            q = self.q_norm(self.to_q(x_chunk))
            if pe_chunk is not None:
                q = apply_rotary_emb(q, pe_chunk, self.rope_type)

            # Full attention with all K/V
            out = self.attention_function(q, k, v, self.heads, mask)
            out = self.to_out(out)
            output_chunks.append(out.cpu())

            del x_chunk, pe_chunk, q, out

        del context_gpu, k, v
        torch.cuda.empty_cache()

        return torch.cat(output_chunks, dim=1)

    def _streamed_attention(
        self,
        q: torch.Tensor,
        K_chunks: list[torch.Tensor],
        V_chunks: list[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute attention with Q on GPU, streaming K/V chunks from CPU.

        Uses online softmax (numerically stable incremental computation) to
        avoid materializing the full attention matrix.

        Args:
            q: [B, Nq, D] query tensor on GPU
            K_chunks: list of [B, Nk_i, D] key chunks on CPU
            V_chunks: list of [B, Nk_i, D] value chunks on CPU
            device: GPU device

        Returns:
            [B, Nq, D] attention output on GPU
        """
        B, Nq, _ = q.shape
        heads = self.heads
        dim_head = self.dim_head

        # Reshape Q for multi-head attention: [B, H, Nq, D]
        q = q.view(B, Nq, heads, dim_head).transpose(1, 2)

        # Initialize online softmax accumulators
        max_scores = torch.full((B, heads, Nq, 1), float('-inf'), device=device, dtype=q.dtype)
        sum_exp = torch.zeros((B, heads, Nq, 1), device=device, dtype=q.dtype)
        output = torch.zeros((B, heads, Nq, dim_head), device=device, dtype=q.dtype)

        scale = dim_head ** -0.5

        for k_chunk, v_chunk in zip(K_chunks, V_chunks):
            # Move this K/V chunk to GPU
            k = k_chunk.to(device, non_blocking=True)
            v = v_chunk.to(device, non_blocking=True)
            torch.cuda.synchronize()

            # Reshape for multi-head: [B, H, Nk, D]
            Nk = k.shape[1]
            k = k.view(B, Nk, heads, dim_head).transpose(1, 2)
            v = v.view(B, Nk, heads, dim_head).transpose(1, 2)

            # Compute attention scores for this K chunk: [B, H, Nq, Nk]
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale

            # Online softmax update (numerically stable)
            chunk_max = scores.max(dim=-1, keepdim=True).values
            new_max = torch.maximum(max_scores, chunk_max)

            # Rescale factors
            exp_diff_old = torch.exp(max_scores - new_max)
            exp_diff_new = torch.exp(chunk_max - new_max)

            # Compute exp(scores - chunk_max) for this chunk
            chunk_exp = torch.exp(scores - chunk_max)
            chunk_sum = chunk_exp.sum(dim=-1, keepdim=True)

            # Update accumulators
            output = output * exp_diff_old + torch.matmul(chunk_exp, v) * exp_diff_new
            sum_exp = sum_exp * exp_diff_old + chunk_sum * exp_diff_new
            max_scores = new_max

            del k, v, scores, chunk_exp, chunk_max, exp_diff_old, exp_diff_new

        # Final normalization
        output = output / (sum_exp + 1e-8)  # Small epsilon for stability

        # Reshape back: [B, Nq, H*D]
        output = output.transpose(1, 2).reshape(B, Nq, heads * dim_head)

        return output
