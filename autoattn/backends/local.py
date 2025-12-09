# autoattn/backends/local.py
"""
Local (Sliding Window) Attention backend.

This provides approximate attention that only attends to nearby tokens,
reducing memory from O(N^2) to O(N * window_size).

Useful for:
- Very long sequences (>8K tokens)
- Memory-constrained scenarios
- When local context is sufficient
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AttentionBackend


class LocalAttention(AttentionBackend):
    """
    Sliding window (local) attention backend.
    
    Each position only attends to tokens within a fixed window,
    providing O(N * W) memory complexity instead of O(N^2).
    
    This is an APPROXIMATE attention - tokens outside the window
    are ignored, which may lose long-range dependencies.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        causal: Whether to apply causal masking (only attend to past)
        window_size: Number of past tokens to attend to
        dropout: Dropout probability (applied during training only)
    
    Input shapes:
        q: [batch, seq_len, d_model]
        k: [batch, seq_len, d_model]
        v: [batch, seq_len, d_model]
    
    Output shape:
        [batch, seq_len, d_model]
    
    Note:
        When causal=True, position t attends to [max(0, t-window_size), t]
        When causal=False, position t attends to [max(0, t-window_size), min(T, t+window_size)]
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        causal: bool = True,
        window_size: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__(d_model, num_heads, causal)
        self.window_size = window_size
        self.dropout = dropout
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute local (sliding window) attention.
        
        Args:
            q: Query tensor [B, T, D]
            k: Key tensor [B, T, D]
            v: Value tensor [B, T, D]
            attn_mask: Optional attention mask (currently ignored for local attention)
        
        Returns:
            Attention output [B, T, D]
        """
        B, T, D = q.shape
        H = self.num_heads
        Dh = self.head_dim
        
        # Reshape: [B, T, D] -> [B, H, T, Dh]
        q = q.view(B, T, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)
        
        # Initialize output
        out = torch.zeros_like(q)
        
        # Dropout probability (only during training)
        dropout_p = self.dropout if self.training else 0.0
        
        # Process each position with its local window
        for t in range(T):
            # Define window boundaries
            start = max(0, t - self.window_size)
            if self.causal:
                end = t + 1  # Only attend to past and current
            else:
                end = min(T, t + self.window_size + 1)  # Bidirectional window
            
            # Extract local keys and values
            q_t = q[:, :, t:t+1, :]           # [B, H, 1, Dh]
            k_w = k[:, :, start:end, :]       # [B, H, W, Dh]
            v_w = v[:, :, start:end, :]       # [B, H, W, Dh]
            
            # Compute attention scores
            scores = torch.matmul(q_t, k_w.transpose(-2, -1)) * self.scale  # [B, H, 1, W]
            
            # Softmax
            attn_weights = F.softmax(scores, dim=-1)
            
            # Apply dropout
            if dropout_p > 0.0:
                attn_weights = F.dropout(attn_weights, p=dropout_p, training=self.training)
            
            # Compute output
            out[:, :, t:t+1, :] = torch.matmul(attn_weights, v_w)
        
        # Reshape back: [B, H, T, Dh] -> [B, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        
        return out
    
    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, causal={self.causal}, dropout={self.dropout}"
        )
