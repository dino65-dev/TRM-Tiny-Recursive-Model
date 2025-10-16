import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """
    SwiGLU: Swish-Gated Linear Unit
    
    Math:
        SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊙ (xV + c)
        where Swish(x) = x * sigmoid(x)
              ⊙ is element-wise product
    
    This is used in the FFN/MLP layers instead of simple ReLU or GELU.
    It has gating mechanism which helps with training.
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Gate projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # Output projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # Value projection
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            output: (batch, seq_len, dim)
        """
        # Swish gate: x * sigmoid(x)
        gate = F.silu(self.w1(x))  # silu = swish = x * sigmoid(x)
        value = self.w3(x)
        # Element-wise product and output projection
        return self.w2(gate * value)