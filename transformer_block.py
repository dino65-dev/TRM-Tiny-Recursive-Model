import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from typing import Optional, Tuple
import numpy as np
from rms_norm import RMSNorm
from mlp_mixer import MLPSequenceMixer
from swiglu import SwiGLU
from mha import MultiHeadAttention

class TransformerBlock(nn.Module):
    """
    Transformer block with either self-attention or MLP sequence mixing.
    
    Structure:
        x = x + Attention/MLPSeq(RMSNorm(x))
        x = x + FFN(RMSNorm(x))
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        use_mlp_mixer: bool = False,
        seq_len: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        self.use_mlp_mixer = use_mlp_mixer
        
        # Pre-normalization
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        
        # Attention or MLP sequence mixing
        if use_mlp_mixer:
            assert seq_len is not None, "seq_len required for MLP mixer"
            self.mixer = MLPSequenceMixer(seq_len)
        else:
            self.mixer = MultiHeadAttention(dim, num_heads, dropout=dropout)
        
        # Feed-forward network with SwiGLU
        self.ffn = SwiGLU(dim, dim * mlp_ratio)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            output: (batch, seq_len, dim)
        """
        # Attention/Mixer with residual
        x = x + self.mixer(self.norm1(x))
        
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        
        return x