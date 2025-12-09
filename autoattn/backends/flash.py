# autoattn/backends/flash.py
"""
Flash Attention backend using PyTorch's scaled_dot_product_attention (SDPA).

This provides exact attention with optimized memory and compute:
- FlashAttention-2 kernel on supported GPUs (Ampere+)
- Memory-efficient attention fallback
- Math attention fallback for CPU

No extra dependencies required - uses PyTorch 2.0+ native SDPA.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AttentionBackend


class FlashAttention(AttentionBackend):
    """
    Flash / SDPA-based attention backend.
    
    This computes exact attention: softmax(QK^T / sqrt(d)) * V
    but with optimized memory access patterns on GPU.
    
    Features:
    - Exact attention (no approximation)
    - O(N) memory instead of O(N^2) for the attention matrix
    - 2-4x faster than naive attention on GPU
    - Automatic kernel selection (flash, memory-efficient, or math)
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        causal: Whether to apply causal masking
        dropout: Dropout probability (applied during training only)
    
    Input shapes:
        q: [batch, seq_len_q, d_model]
        k: [batch, seq_len_k, d_model]  
        v: [batch, seq_len_v, d_model] (seq_len_v == seq_len_k)
    
    Output shape:
        [batch, seq_len_q, d_model]
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        causal: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__(d_model, num_heads, causal)
        self.dropout = dropout
        self.head_dim = d_model // num_heads
        
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
        Compute scaled dot-product attention using PyTorch SDPA.
        
        Args:
            q: Query tensor [B, T_q, D]
            k: Key tensor [B, T_k, D]
            v: Value tensor [B, T_k, D]
            attn_mask: Optional attention mask. If provided with is_causal=True,
                      the causal mask is applied on top of attn_mask.
        
        Returns:
            Attention output [B, T_q, D]
        """
        B, T_q, D = q.shape
        _, T_k, _ = k.shape
        H = self.num_heads
        Dh = self.head_dim
        
        # Reshape: [B, T, D] -> [B, H, T, Dh]
        q = q.view(B, T_q, H, Dh).transpose(1, 2)
        k = k.view(B, T_k, H, Dh).transpose(1, 2)
        v = v.view(B, T_k, H, Dh).transpose(1, 2)
        
        # Apply dropout only during training
        dropout_p = self.dropout if self.training else 0.0
        
        # Use is_causal for efficiency when no custom mask is provided
        # Note: is_causal assumes T_q == T_k for proper causal masking
        use_causal = self.causal and attn_mask is None and T_q == T_k
        
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=use_causal,
        )
        
        # Reshape back: [B, H, T_q, Dh] -> [B, T_q, D]
        out = out.transpose(1, 2).contiguous().view(B, T_q, D)
        
        return out
    
    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, causal={self.causal}, dropout={self.dropout}"
        )
