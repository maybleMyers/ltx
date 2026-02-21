import torch

from ltx_core.model.transformer.gelu_approx import GELUApprox


class FeedForward(torch.nn.Module):
    def __init__(self, dim: int, dim_out: int, mult: int = 4) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        project_in = GELUApprox(dim, inner_dim)

        self.net = torch.nn.Sequential(project_in, torch.nn.Identity(), torch.nn.Linear(inner_dim, dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward_chunked(self, x: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
        """
        Process FFN in chunks along the sequence dimension to reduce peak memory.

        This is mathematically equivalent to forward() but processes the sequence
        in smaller chunks, reducing the peak memory from the 4x expansion in FFN.
        For long videos (1000+ frames), this can reduce peak FFN memory by 50-75%.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            chunk_size: Number of tokens to process at once (default 4096)

        Returns:
            Output tensor of same shape as input
        """
        if x.shape[1] <= chunk_size:
            return self.forward(x)

        batch, seq_len, dim = x.shape
        output = torch.empty_like(x)

        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            output[:, start:end, :] = self.net(x[:, start:end, :])

        return output


class TensorParallelFeedForward(torch.nn.Module):
    """
    FFN split across two GPUs to halve peak memory per device.

    Instead of allocating a single (batch, seq_len, inner_dim) tensor on one GPU,
    this splits inner_dim in half across two GPUs. Each GPU computes its half
    independently, then results are gathered and summed.

    This eliminates the need for FFN chunking on memory-constrained setups.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        mult: int = 4,
        device0: torch.device = None,
        device1: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        half_inner = inner_dim // 2

        self.device0 = device0 or torch.device("cuda:0")
        self.device1 = device1 or torch.device("cuda:1")
        self.half_inner = half_inner
        dtype = dtype or torch.bfloat16  # Default to bfloat16 for LTX models

        # GPU:0 handles first half of inner_dim
        # Note: device placement is handled after weight copying in from_feed_forward()
        self.proj_in_0 = torch.nn.Linear(dim, half_inner, dtype=dtype)
        self.proj_out_0 = torch.nn.Linear(half_inner, dim_out, dtype=dtype)

        # GPU:1 handles second half of inner_dim
        self.proj_in_1 = torch.nn.Linear(dim, half_inner, dtype=dtype)
        self.proj_out_1 = torch.nn.Linear(half_inner, dim_out, dtype=dtype)

    def _get_device_restricted_modules(self) -> set:
        """
        Return modules that should NOT be moved by weighs_to_device or other bulk move operations.

        These modules must stay on device1 for tensor parallelism to work correctly.
        """
        return {self.proj_in_1, self.proj_out_1}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Safety check: ensure device1 modules are on device1
        # (PyTorch's module system can sometimes move them despite our _apply override)
        if self.proj_in_1.weight.device != self.device1:
            self.proj_in_1 = self.proj_in_1.to(self.device1)
            self.proj_out_1 = self.proj_out_1.to(self.device1)

        # x is expected on device0
        # Send copy to GPU:1 (non-blocking to overlap with GPU:0 computation)
        x1 = x.to(self.device1, non_blocking=True)

        # Parallel computation on both GPUs
        # GPU:0 computes first half of hidden dim
        h0 = torch.nn.functional.gelu(self.proj_in_0(x), approximate="tanh")
        # GPU:1 computes second half of hidden dim
        h1 = torch.nn.functional.gelu(self.proj_in_1(x1), approximate="tanh")

        # Project back to output dim
        out0 = self.proj_out_0(h0)
        out1 = self.proj_out_1(h1)

        # Gather: bring GPU:1 result back to GPU:0 and sum
        # The sum is correct because we split the hidden dim, so each GPU
        # computes a partial contribution to the final output
        out1_on_0 = out1.to(self.device0, non_blocking=True)
        torch.cuda.synchronize(self.device1)  # Ensure transfer complete

        return out0 + out1_on_0

    def forward_chunked(self, x: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
        """Chunked forward for compatibility - uses tensor parallel within each chunk."""
        if x.shape[1] <= chunk_size:
            return self.forward(x)

        batch, seq_len, dim = x.shape
        # Allocate output on device0
        output = torch.empty(batch, seq_len, self.proj_out_0.out_features,
                            device=self.device0, dtype=x.dtype)

        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            output[:, start:end, :] = self.forward(x[:, start:end, :])

        return output

    def _apply(self, fn):
        """
        Override _apply to only apply to device0 submodules.

        PyTorch's Module.to() uses _apply() internally, which would normally
        move ALL submodules. We override this to only move device0 submodules,
        keeping device1 submodules permanently on device1.
        """
        # Apply to device0 submodules only
        self.proj_in_0._apply(fn)
        self.proj_out_0._apply(fn)
        # DO NOT apply to device1 submodules - they must stay on device1

        # Update device0 tracking if parameters moved
        if hasattr(self.proj_in_0, 'weight') and self.proj_in_0.weight is not None:
            self.device0 = self.proj_in_0.weight.device

        return self

    def to(self, *args, **kwargs):
        """
        Override to() to only move device0 submodules.

        This handles explicit .to() calls while _apply handles internal moves.
        """
        # Extract device from args/kwargs
        device = None
        dtype = None

        if args:
            if isinstance(args[0], (torch.device, str)):
                device = args[0]
            elif isinstance(args[0], torch.dtype):
                dtype = args[0]
        device = kwargs.get('device', device)
        dtype = kwargs.get('dtype', dtype)

        if device is not None:
            device = torch.device(device) if isinstance(device, str) else device
            self.proj_in_0 = self.proj_in_0.to(device)
            self.proj_out_0 = self.proj_out_0.to(device)
            self.device0 = device

        if dtype is not None:
            # Apply dtype to all modules
            self.proj_in_0 = self.proj_in_0.to(dtype=dtype)
            self.proj_out_0 = self.proj_out_0.to(dtype=dtype)
            self.proj_in_1 = self.proj_in_1.to(dtype=dtype)
            self.proj_out_1 = self.proj_out_1.to(dtype=dtype)

        return self

    def cuda(self, device=None):
        """Override cuda() to only move device0 submodules."""
        if device is None:
            device = self.device0
        return self.to(device=device)

    def cpu(self):
        """Override cpu() - moves device0 parts to CPU, keeps device1 on GPU."""
        self.proj_in_0 = self.proj_in_0.cpu()
        self.proj_out_0 = self.proj_out_0.cpu()
        self.device0 = torch.device("cpu")
        return self

    @classmethod
    def from_feed_forward(
        cls,
        ff: "FeedForward",
        device0: torch.device,
        device1: torch.device,
    ) -> "TensorParallelFeedForward":
        """
        Convert existing FeedForward to tensor-parallel version.

        Copies weights from the original module, splitting the inner dimension
        across two devices.

        Args:
            ff: Original FeedForward module
            device0: Primary GPU (keeps first half of weights)
            device1: Secondary GPU (keeps second half of weights)

        Returns:
            New TensorParallelFeedForward with copied weights
        """
        # Extract dimensions and dtype from existing module
        proj_in = ff.net[0]  # GELUApprox
        proj_out = ff.net[2]  # Linear

        dim = proj_in.proj.in_features
        dim_out = proj_out.out_features
        inner_dim = proj_in.proj.out_features
        mult = inner_dim // dim
        half_inner = inner_dim // 2
        dtype = proj_in.proj.weight.dtype  # Preserve original dtype

        # Create new module with correct dtype
        tp_ff = cls(dim, dim_out, mult, device0, device1, dtype=dtype)

        # Copy weights from original, splitting inner_dim
        with torch.no_grad():
            # First projection (dim -> inner_dim): split output dim
            # Original weight shape: (inner_dim, dim)
            # Original bias shape: (inner_dim,)
            tp_ff.proj_in_0.weight.copy_(proj_in.proj.weight[:half_inner])
            tp_ff.proj_in_0.bias.copy_(proj_in.proj.bias[:half_inner])
            tp_ff.proj_in_1.weight.copy_(proj_in.proj.weight[half_inner:])
            tp_ff.proj_in_1.bias.copy_(proj_in.proj.bias[half_inner:])

            # Second projection (inner_dim -> dim_out): split input dim
            # Original weight shape: (dim_out, inner_dim)
            # Original bias shape: (dim_out,)
            tp_ff.proj_out_0.weight.copy_(proj_out.weight[:, :half_inner])
            tp_ff.proj_out_1.weight.copy_(proj_out.weight[:, half_inner:])
            # Bias: split equally since we're summing the outputs
            tp_ff.proj_out_0.bias.copy_(proj_out.bias / 2)
            tp_ff.proj_out_1.bias.copy_(proj_out.bias / 2)

        # Explicitly move submodules to their target devices
        # (Linear creation with device= doesn't always work correctly with copy_)
        tp_ff.proj_in_0 = tp_ff.proj_in_0.to(device0)
        tp_ff.proj_out_0 = tp_ff.proj_out_0.to(device0)
        tp_ff.proj_in_1 = tp_ff.proj_in_1.to(device1)
        tp_ff.proj_out_1 = tp_ff.proj_out_1.to(device1)

        return tp_ff
