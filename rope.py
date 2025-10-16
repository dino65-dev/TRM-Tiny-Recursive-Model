import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from typing import Optional, Tuple
import numpy as np

class RotaryEmbedding(nn.Module):
    """
    RoPE: Rotary Position Embedding
    Applies rotation to query/key vectors based on position.
    
    Math for position m and dimension 2i, 2i+1:
        θ_i = 10000^(-2i/d)
        
        [q_{2i}  ]   [cos(m*θ_i)  -sin(m*θ_i)] [q_{2i}  ]
        [q_{2i+1}] = [sin(m*θ_i)   cos(m*θ_i)] [q_{2i+1}]
    
    This encodes position through rotation in 2D subspaces.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Compute theta for each dimension pair
        # θ_i = base^(-2i/d) for i = 0, 1, ..., d/2-1
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute position embeddings
        self._init_cos_sin_cache(max_seq_len)
        
    def _init_cos_sin_cache(self, seq_len: int):
        """Precompute cos and sin for all positions"""
        # positions: [0, 1, 2, ..., seq_len-1]
        position = torch.arange(seq_len).float()
        # freqs: (seq_len, dim//2) - outer product of positions and inv_freq
        freqs = torch.einsum('i,j->ij', position, self.inv_freq)
        # emb: (seq_len, dim) - interleave to match [x0, x1, x2, x3, ...]
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
        
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate pairs: [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]
        This implements the 2D rotation matrix multiplication efficiently.
        """
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Apply rotary embeddings to input tensor.
        
        Args:
            x: (batch, seq_len, dim)
            seq_len: sequence length
        Returns:
            x_rotated: (batch, seq_len, dim)
        """
        cos = self.cos_cached[:seq_len, :]
        sin = self.sin_cached[:seq_len, :]
        
        # Apply rotation: x*cos + rotate_half(x)*sin
        return x * cos + self.rotate_half(x) * sin