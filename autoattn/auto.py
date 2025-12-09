from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .backends.dense import DenseAttention
from .backends.flash import FlashAttention
from .backends.local import LocalAttention


@dataclass(frozen=True)
class _ProblemSignature:
    """
    Compact signature describing a "regime" in which a backend choice is valid.

    We bucket sequence length etc. so we don't explode the cache.
    """
    seq_bucket: int
    d_model_bucket: int
    batch_bucket: int
    device_type: str  # "cpu" or "cuda"
    phase: str        # "train" or "inference"
    mode: str         # "auto" | "performance" | "memory"


class AutoAttention(nn.Module):
    """
    Router module that picks an attention backend (dense, flash, local, ...)
    based on input shape, device, and user preferences.

    It guarantees correctness by always having DenseAttention as a fallback.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        causal: bool = True,
        mode: str = "auto",  # "auto", "performance", "memory"
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        window_size: int = 512,  # for local attention
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.causal = causal
        self.mode = mode
        self.dropout = dropout

        # This is just a default; in forward we always read q.device instead.
        self._default_device = device

        # Register available backends
        backends = {}

        # Dense attention - always available as fallback (includes learned projections)
        backends["dense"] = DenseAttention(d_model, num_heads, causal, dropout)

        # Flash attention - uses PyTorch's SDPA, available on PyTorch 2.0+
        backends["flash"] = FlashAttention(d_model, num_heads, causal, dropout)

        # Local/sparse attention - sliding window
        backends["local"] = LocalAttention(d_model, num_heads, causal, window_size, dropout)

        self.backends = nn.ModuleDict(backends)

        # Cache: ProblemSignature -> backend_name
        self._choice_cache: Dict[_ProblemSignature, str] = {}

    # --------- helpers ---------

    @staticmethod
    def _bucket(value: int) -> int:
        """
        Bucket an integer into powers-of-two buckets to avoid overfitting the cache
        to exact sequence lengths / batch sizes.
        """
        if value <= 16:
            return 16
        # round to nearest power of 2
        import math

        return 2 ** int(round(math.log2(value)))

    def _make_signature(
        self,
        q: torch.Tensor,
        metadata: Optional[dict],
    ) -> _ProblemSignature:
        bsz, seq_len, d_model = q.shape

        seq_bucket = self._bucket(seq_len)
        d_model_bucket = self._bucket(d_model)
        batch_bucket = self._bucket(bsz)

        device_type = "cuda" if q.is_cuda else "cpu"

        # training phase: if module is in .train(), treat as "train"
        phase = metadata.get("phase") if metadata and "phase" in metadata else (
            "train" if self.training else "inference"
        )

        mode = metadata.get("mode") if metadata and "mode" in metadata else self.mode

        return _ProblemSignature(
            seq_bucket=seq_bucket,
            d_model_bucket=d_model_bucket,
            batch_bucket=batch_bucket,
            device_type=device_type,
            phase=phase,
            mode=mode,
        )

    def _choose_backend_name(self, sig: _ProblemSignature) -> str:
        """
        Select the best backend based on the problem signature.
        
        Routing logic:
        - CPU: always use dense (most compatible)
        - GPU + short sequences (≤2048): use flash (fast, exact)
        - GPU + very long sequences (>4096): use local (memory efficient, approximate)
        - Default: dense (correctness fallback)
        """
        # CPU: dense is the safest choice
        if sig.device_type == "cpu":
            return "dense"

        # Performance mode on GPU: prefer flash for shorter sequences
        if sig.mode in ("auto", "performance"):
            if "flash" in self.backends and sig.seq_bucket <= 2048:
                return "flash"

        # Memory mode or very long sequences: use local attention
        if sig.mode == "memory" or sig.seq_bucket > 4096:
            if "local" in self.backends:
                return "local"

        # Default fallback
        return "dense"

    def _get_backend(
        self,
        q: torch.Tensor,
        metadata: Optional[dict],
    ) -> nn.Module:
        sig = self._make_signature(q, metadata)

        if sig in self._choice_cache:
            backend_name = self._choice_cache[sig]
            return self.backends[backend_name]

        backend_name = self._choose_backend_name(sig)
        self._choice_cache[sig] = backend_name
        return self.backends[backend_name]

    def get_backend_name(
        self,
        q: torch.Tensor,
        metadata: Optional[dict] = None,
    ) -> str:
        """Public method to inspect which backend would be chosen."""
        sig = self._make_signature(q, metadata)
        return self._choose_backend_name(sig)

    # --------- public forward ---------

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        metadata: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        q, k, v: [batch, seq_len, d_model]
        attn_mask: optional mask, passed through to backend.
        metadata: optional dict with hints, e.g.
            {
              "phase": "train" | "inference",
              "mode": "performance" | "memory" | "auto",
              "task": "causal_lm" | "seq2seq" | ...
            }
        """
        backend = self._get_backend(q, metadata)
        return backend(q, k, v, attn_mask=attn_mask)
