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
