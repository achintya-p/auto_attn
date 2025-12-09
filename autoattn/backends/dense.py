from typing import Optional

import torch
import torch.nn as nn

from .base import AttentionBackend


class DenseAttention(AttentionBackend):
    """
    Vanilla multi-head attention using PyTorch's MultiheadAttention.

    This is the correctness fallback: it should always work, on CPU or GPU.
    """

    def __init__(self, d_model: int, num_heads: int, causal: bool = True):
        super().__init__(d_model, num_heads, causal)

        # batch_first=True so we can use [B, T, D]
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
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
        q, k, v: [batch, seq_len, d_model]
        attn_mask: optional, broadcastable mask.
        If causal=True and no mask is provided, we build a causal mask.
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
