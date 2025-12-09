# autoattn/backends/flash.py

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AttentionBackend


class FlashAttention(AttentionBackend):
    """
    Flash / SDPA-based attention backend.
    This is exact attention, just faster on GPU.
    """

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # PyTorch expects [B, H, T, D_head]
        B, T, D = q.shape
        H = self.num_heads
        Dh = D // H

        q = q.view(B, T, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=self.causal
        )

        # Back to [B, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return out
