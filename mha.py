import torch
import torch.nn as nn
import torch.nn.functional as F
from rope import RotaryEmbedding

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with RoPE.
    
    Math:
        Q = XW_Q, K = XW_K, V = XW_V
        Q, K = RoPE(Q), RoPE(K)
        
        Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
        
        Multi-head: split into h heads, concatenate results
    """
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        max_seq_len: int = 2048,
        dropout: float = 0.0
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1/sqrt(d_k) for scaled dot-product
        
        # Q, K, V projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            output: (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, dim)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head: (batch, seq_len, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply RoPE to Q and K
        # Shape: (batch, seq_len, num_heads, head_dim)
        q = self.rope(q, seq_len)
        k = self.rope(k, seq_len)
        
        # Transpose for attention: (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        # QK^T: (batch, num_heads, seq_len, seq_len)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # (batch, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)
        
        # Concatenate heads: (batch, seq_len, dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.dim)
        
        # Final projection
        output = self.o_proj(attn_output)
        return output
