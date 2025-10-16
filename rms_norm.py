import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    RMSNorm: y = (x / RMS(x)) * weight
    where RMS(x) = sqrt(mean(x^2) + eps)
    
    Unlike LayerNorm which uses mean and variance, RMSNorm only uses RMS.
    This is simpler and often performs better.
    
    Math:
        RMS(x) = sqrt( (1/d) * sum(x_i^2) + eps )
        output = (x / RMS(x)) * learnable_weight
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        # Learnable scale parameter (one per dimension)
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            normalized: (batch, seq_len, dim)
        """
        # Compute RMS over last dimension
        # x^2: element-wise square
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        x_normalized = x / rms
        return x_normalized * self.weight