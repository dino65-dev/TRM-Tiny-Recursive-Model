import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from typing import Optional, Tuple
import numpy as np

class MLPSequenceMixer(nn.Module):
    """
    MLP-Mixer style token mixing (sequence mixing).
    
    Math:
        For input X ∈ R^(L×D):
        1. Transpose to X^T ∈ R^(D×L)
        2. Apply MLP along L dimension: W_2 * GeLU(W_1 * X^T)
        3. Transpose back
        
    This is content-independent mixing (weights are fixed, not data-dependent).
    Works well for small, fixed sequence lengths.
    
    W_1 ∈ R^(L×L), W_2 ∈ R^(L×L)
    Same W_1, W_2 applied to all D channels independently.
    """
    def __init__(self, seq_len: int, hidden_factor: int = 4):
        super().__init__()
        self.seq_len = seq_len
        hidden_dim = seq_len * hidden_factor
        
        # Token mixing MLPs (applied along sequence dimension)
        self.mlp1 = nn.Linear(seq_len, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, seq_len)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            output: (batch, seq_len, dim)
        """
        # Transpose: (batch, dim, seq_len)
        x_t = x.transpose(1, 2)
        
        # Apply MLP along sequence dimension
        # This mixes information across positions
        x_mixed = self.mlp1(x_t)  # (batch, dim, hidden_dim)
        x_mixed = self.activation(x_mixed)
        x_mixed = self.mlp2(x_mixed)  # (batch, dim, seq_len)
        
        # Transpose back: (batch, seq_len, dim)
        return x_mixed.transpose(1, 2)