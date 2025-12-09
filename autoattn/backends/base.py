from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


"""
vanilla fallback attention
"""

class AttentionBackend(ABC, nn.Module):
    """
    Abstract base class for all attention backends.

    Each backend must implement:
        forward(q, k, v, attn_mask=None) -> Tensor

    Shapes:
        q, k, v: [batch, seq_len, d_model]
        attn_mask: optional, broadcastable to [batch, seq_len, seq_len]
    """

    def __init__(self, d_model: int, num_heads: int, causal: bool = True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.causal = causal

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError
