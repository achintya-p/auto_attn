from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AttentionBackend


class LocalAttention(AttentionBackend): 
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        causal: bool = True,
        window_size: int = 512
    ):
        super().__init__(d_model, num_heads, causal)
        self.window_size = window_size
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        B, T, D = q.shape
        H = self.num_heads
        Dh = D // H

        q = q.view(B, T, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)

        out = torch.zeros_like(q)

        for t in range(T):
            start = max(0, t - self.window_size)
            end = t + 1 if self.causal else min(T, t + self.window_size)

            q_t = q[:, :, t:t+1, :]
            k_w = k[:, :, start:end, :]
            v_w = v[:, :, start:end, :]

            scores = torch.matmul(q_t, k_w.transpose(-2, -1)) / (Dh ** 0.5)
            attn = F.softmax(scores, dim=-1)
            out[:, :, t:t+1, :] = torch.matmul(attn, v_w)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return out