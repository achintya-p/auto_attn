# autoattn/backends/dense.py
"""
Dense (Vanilla) Attention backend using PyTorch's MultiheadAttention.

This is the correctness fallback - always works on CPU or GPU,
uses well-tested PyTorch implementation with learned projections.

Note: This backend includes learned W_q, W_k, W_v, W_o projection matrices,
unlike Flash and Local which compute raw attention on inputs directly.
"""

from typing import Optional

import torch
import torch.nn as nn

from .base import AttentionBackend


class DenseAttention(AttentionBackend):
    """
    Vanilla multi-head attention using PyTorch's MultiheadAttention.
    
    This is the correctness fallback: it should always work, on CPU or GPU.
    Uses PyTorch's highly optimized and well-tested implementation.
    
    IMPORTANT: Unlike Flash/Local backends, this includes learned projection
    matrices (W_q, W_k, W_v, W_o). The other backends compute raw attention:
        softmax(QK^T / sqrt(d)) * V
    
    While this backend computes:
        softmax((QW_q)(KW_k)^T / sqrt(d)) * (VW_v) * W_o
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        causal: Whether to apply causal masking
        dropout: Dropout probability
    
    Input shapes:
        q, k, v: [batch, seq_len, d_model]
    
    Output shape:
        [batch, seq_len, d_model]
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        causal: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__(d_model, num_heads, causal)
        
        # batch_first=True so we can use [B, T, D]
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute multi-head attention.
        
        Args:
            q: Query tensor [B, T_q, D]
            k: Key tensor [B, T_k, D]
            v: Value tensor [B, T_k, D]
            attn_mask: Optional attention mask
        
        Returns:
            Attention output [B, T_q, D]
        """
        bsz, tgt_len, _ = q.shape
        _, src_len, _ = k.shape
        
        # Build a simple causal mask if needed and none was provided.
        # MultiheadAttention uses a float mask with -inf for disallowed positions.
        if self.causal and attn_mask is None:
            # [tgt_len, src_len]
            causal_mask = torch.full(
                (tgt_len, src_len),
                float("-inf"),
                device=q.device,
                dtype=q.dtype,
            )
            causal_mask = torch.triu(causal_mask, diagonal=1)
            attn_mask = causal_mask  # MHA expects [T, S] or [N*num_heads, T, S]
        
        out, _ = self.attn(
            q,  # [B, T, D]
            k,
            v,
            attn_mask=attn_mask,
            need_weights=False,
        )
        
        return out
    
    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, num_heads={self.num_heads}, causal={self.causal}"
