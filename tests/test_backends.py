"""
Unit tests for attention backends.

Tests verify:
1. Output shapes are correct
2. Causal masking works (future tokens don't affect past)
3. Outputs are numerically reasonable (no NaNs, bounded values)
4. Backends produce similar results for same inputs (correctness)
"""

import pytest
import torch
import torch.nn as nn

from autoattn.backends.dense import DenseAttention
from autoattn.backends.flash import FlashAttention
from autoattn.backends.local import LocalAttention


# Test fixtures
@pytest.fixture
def default_config():
    return {
        "d_model": 64,
        "num_heads": 4,
        "batch_size": 2,
        "seq_len": 32,
    }


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_qkv(batch_size: int, seq_len: int, d_model: int, device: torch.device):
    """Create random Q, K, V tensors."""
    torch.manual_seed(42)  # reproducibility
    q = torch.randn(batch_size, seq_len, d_model, device=device)
    k = torch.randn(batch_size, seq_len, d_model, device=device)
    v = torch.randn(batch_size, seq_len, d_model, device=device)
    return q, k, v


# ============================================================================
# DenseAttention Tests
# ============================================================================

class TestDenseAttention:
    """Tests for DenseAttention backend."""

    def test_output_shape(self, default_config, device):
        """Output should have same shape as input Q."""
        cfg = default_config
        attn = DenseAttention(cfg["d_model"], cfg["num_heads"], causal=True).to(device)
        q, k, v = make_qkv(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], device)

        out = attn(q, k, v)

        assert out.shape == q.shape, f"Expected {q.shape}, got {out.shape}"

    def test_no_nans(self, default_config, device):
        """Output should not contain NaN values."""
        cfg = default_config
        attn = DenseAttention(cfg["d_model"], cfg["num_heads"], causal=True).to(device)
        q, k, v = make_qkv(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], device)

        out = attn(q, k, v)

        assert not torch.isnan(out).any(), "Output contains NaN values"

    def test_bounded_output(self, default_config, device):
        """Output values should be bounded (no explosion)."""
        cfg = default_config
        attn = DenseAttention(cfg["d_model"], cfg["num_heads"], causal=True).to(device)
        q, k, v = make_qkv(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], device)

        out = attn(q, k, v)

        # Output should be roughly in same range as V (attention is a weighted avg)
        assert out.abs().max() < 100, f"Output values too large: {out.abs().max()}"

    def test_causal_masking(self, default_config, device):
        """Changing future tokens should not affect past outputs."""
        cfg = default_config
        attn = DenseAttention(cfg["d_model"], cfg["num_heads"], causal=True).to(device)
        q, k, v = make_qkv(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], device)

        # Get output with original inputs
        out1 = attn(q.clone(), k.clone(), v.clone())

        # Modify future tokens (last 10 positions)
        k_modified = k.clone()
        v_modified = v.clone()
        k_modified[:, -10:, :] = torch.randn_like(k_modified[:, -10:, :])
        v_modified[:, -10:, :] = torch.randn_like(v_modified[:, -10:, :])

        out2 = attn(q.clone(), k_modified, v_modified)

        # First N-10 positions should be identical (they can't see future)
        early_positions = cfg["seq_len"] - 10
        assert torch.allclose(
            out1[:, :early_positions, :],
            out2[:, :early_positions, :],
            atol=1e-5
        ), "Causal masking failed: past outputs changed when future tokens modified"

    def test_non_causal_mode(self, default_config, device):
        """Non-causal attention should allow bidirectional attention."""
        cfg = default_config
        attn = DenseAttention(cfg["d_model"], cfg["num_heads"], causal=False).to(device)
        q, k, v = make_qkv(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], device)

        out = attn(q, k, v)

        assert out.shape == q.shape

    def test_different_seq_lengths(self, device):
        """Test with various sequence lengths."""
        d_model, num_heads = 64, 4
        attn = DenseAttention(d_model, num_heads, causal=True).to(device)

        for seq_len in [1, 16, 64, 128]:
            q, k, v = make_qkv(2, seq_len, d_model, device)
            out = attn(q, k, v)
            assert out.shape == (2, seq_len, d_model)


# ============================================================================
# FlashAttention Tests
# ============================================================================

class TestFlashAttention:
    """Tests for FlashAttention (SDPA) backend."""

    def test_output_shape(self, default_config, device):
        """Output should have same shape as input Q."""
        cfg = default_config
        attn = FlashAttention(cfg["d_model"], cfg["num_heads"], causal=True).to(device)
        q, k, v = make_qkv(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], device)

        out = attn(q, k, v)

        assert out.shape == q.shape, f"Expected {q.shape}, got {out.shape}"

    def test_no_nans(self, default_config, device):
        """Output should not contain NaN values."""
        cfg = default_config
        attn = FlashAttention(cfg["d_model"], cfg["num_heads"], causal=True).to(device)
        q, k, v = make_qkv(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], device)

        out = attn(q, k, v)

        assert not torch.isnan(out).any(), "Output contains NaN values"

    def test_bounded_output(self, default_config, device):
        """Output values should be bounded."""
        cfg = default_config
        attn = FlashAttention(cfg["d_model"], cfg["num_heads"], causal=True).to(device)
        q, k, v = make_qkv(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], device)

        out = attn(q, k, v)

        assert out.abs().max() < 100, f"Output values too large: {out.abs().max()}"

    def test_causal_masking(self, default_config, device):
        """Changing future tokens should not affect past outputs."""
        cfg = default_config
        attn = FlashAttention(cfg["d_model"], cfg["num_heads"], causal=True).to(device)
        q, k, v = make_qkv(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], device)

        out1 = attn(q.clone(), k.clone(), v.clone())

        k_modified = k.clone()
        v_modified = v.clone()
        k_modified[:, -10:, :] = torch.randn_like(k_modified[:, -10:, :])
        v_modified[:, -10:, :] = torch.randn_like(v_modified[:, -10:, :])

        out2 = attn(q.clone(), k_modified, v_modified)

        early_positions = cfg["seq_len"] - 10
        assert torch.allclose(
            out1[:, :early_positions, :],
            out2[:, :early_positions, :],
            atol=1e-5
        ), "Causal masking failed"

    def test_deterministic_output(self, default_config, device):
        """Flash attention should produce deterministic results for same input."""
        cfg = default_config
        flash = FlashAttention(cfg["d_model"], cfg["num_heads"], causal=True).to(device)

        q, k, v = make_qkv(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], device)

        out1 = flash(q.clone(), k.clone(), v.clone())
        out2 = flash(q.clone(), k.clone(), v.clone())

        # Same input should produce identical output
        assert torch.allclose(out1, out2, atol=1e-6), \
            f"Non-deterministic output, max diff: {(out1 - out2).abs().max()}"

    def test_different_seq_lengths(self, device):
        """Test with various sequence lengths."""
        d_model, num_heads = 64, 4
        attn = FlashAttention(d_model, num_heads, causal=True).to(device)

        for seq_len in [1, 16, 64, 128, 256]:
            q, k, v = make_qkv(2, seq_len, d_model, device)
            out = attn(q, k, v)
            assert out.shape == (2, seq_len, d_model)


# ============================================================================
# LocalAttention Tests
# ============================================================================

class TestLocalAttention:
    """Tests for LocalAttention (sliding window) backend."""

    def test_output_shape(self, default_config, device):
        """Output should have same shape as input Q."""
        cfg = default_config
        attn = LocalAttention(
            cfg["d_model"], cfg["num_heads"], causal=True, window_size=16
        ).to(device)
        q, k, v = make_qkv(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], device)

        out = attn(q, k, v)

        assert out.shape == q.shape, f"Expected {q.shape}, got {out.shape}"

    def test_no_nans(self, default_config, device):
        """Output should not contain NaN values."""
        cfg = default_config
        attn = LocalAttention(
            cfg["d_model"], cfg["num_heads"], causal=True, window_size=16
        ).to(device)
        q, k, v = make_qkv(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], device)

        out = attn(q, k, v)

        assert not torch.isnan(out).any(), "Output contains NaN values"

    def test_bounded_output(self, default_config, device):
        """Output values should be bounded."""
        cfg = default_config
        attn = LocalAttention(
            cfg["d_model"], cfg["num_heads"], causal=True, window_size=16
        ).to(device)
        q, k, v = make_qkv(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], device)

        out = attn(q, k, v)

        assert out.abs().max() < 100, f"Output values too large: {out.abs().max()}"

    def test_causal_masking(self, default_config, device):
        """Local attention with causal=True should not see future."""
        cfg = default_config
        attn = LocalAttention(
            cfg["d_model"], cfg["num_heads"], causal=True, window_size=16
        ).to(device)
        q, k, v = make_qkv(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], device)

        out1 = attn(q.clone(), k.clone(), v.clone())

        # Modify future tokens
        k_modified = k.clone()
        v_modified = v.clone()
        k_modified[:, -10:, :] = torch.randn_like(k_modified[:, -10:, :])
        v_modified[:, -10:, :] = torch.randn_like(v_modified[:, -10:, :])

        out2 = attn(q.clone(), k_modified, v_modified)

        # Early positions should be identical
        early_positions = cfg["seq_len"] - 10
        assert torch.allclose(
            out1[:, :early_positions, :],
            out2[:, :early_positions, :],
            atol=1e-5
        ), "Causal masking failed"

    def test_window_size_effect(self, device):
        """Tokens outside window should not affect output."""
        d_model, num_heads = 64, 4
        window_size = 8
        seq_len = 32

        attn = LocalAttention(d_model, num_heads, causal=True, window_size=window_size).to(device)
        q, k, v = make_qkv(2, seq_len, d_model, device)

        out1 = attn(q.clone(), k.clone(), v.clone())

        # Modify tokens far in the past (outside window)
        k_modified = k.clone()
        v_modified = v.clone()
        k_modified[:, :5, :] = torch.randn_like(k_modified[:, :5, :])
        v_modified[:, :5, :] = torch.randn_like(v_modified[:, :5, :])

        out2 = attn(q.clone(), k_modified, v_modified)

        # Positions far from the modified region should be identical
        # Position 20+ is at least 15 steps away from positions 0-4
        # With window_size=8, position 20 only sees positions 12-20
        assert torch.allclose(
            out1[:, 20:, :],
            out2[:, 20:, :],
            atol=1e-5
        ), "Window size not respected"

    def test_large_window_matches_flash(self, default_config, device):
        """With window >= seq_len, local should match flash (both compute raw attention)."""
        cfg = default_config
        seq_len = cfg["seq_len"]

        # Flash and Local both compute raw attention (no learned projections)
        flash = FlashAttention(cfg["d_model"], cfg["num_heads"], causal=True).to(device)
        local = LocalAttention(
            cfg["d_model"], cfg["num_heads"], causal=True, window_size=seq_len
        ).to(device)

        q, k, v = make_qkv(cfg["batch_size"], seq_len, cfg["d_model"], device)

        out_flash = flash(q, k, v)
        out_local = local(q, k, v)

        # With window covering full sequence, local should be equivalent to flash
        assert torch.allclose(out_flash, out_local, atol=1e-4), \
            f"Max diff: {(out_flash - out_local).abs().max()}"

    def test_different_window_sizes(self, device):
        """Test various window sizes."""
        d_model, num_heads, seq_len = 64, 4, 64

        for window_size in [4, 16, 32, 64]:
            attn = LocalAttention(d_model, num_heads, causal=True, window_size=window_size).to(device)
            q, k, v = make_qkv(2, seq_len, d_model, device)
            out = attn(q, k, v)
            assert out.shape == (2, seq_len, d_model)
            assert not torch.isnan(out).any()


# ============================================================================
# Cross-Backend Comparison Tests
# ============================================================================

class TestBackendConsistency:
    """Tests comparing outputs across backends."""

    def test_all_backends_same_shape(self, default_config, device):
        """All backends should produce same output shape."""
        cfg = default_config
        q, k, v = make_qkv(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], device)

        backends = [
            DenseAttention(cfg["d_model"], cfg["num_heads"], causal=True).to(device),
            FlashAttention(cfg["d_model"], cfg["num_heads"], causal=True).to(device),
            LocalAttention(cfg["d_model"], cfg["num_heads"], causal=True, window_size=cfg["seq_len"]).to(device),
        ]

        shapes = [backend(q.clone(), k.clone(), v.clone()).shape for backend in backends]

        assert all(s == shapes[0] for s in shapes), f"Shape mismatch: {shapes}"

    def test_flash_local_equivalence(self, default_config, device):
        """Flash and Local (with full window) should be numerically equivalent.
        
        Note: Dense uses nn.MultiheadAttention with learned projections,
        while Flash and Local compute raw attention. So we compare Flash vs Local.
        """
        cfg = default_config
        seq_len = cfg["seq_len"]

        flash = FlashAttention(cfg["d_model"], cfg["num_heads"], causal=True).to(device)
        local = LocalAttention(
            cfg["d_model"], cfg["num_heads"], causal=True, window_size=seq_len
        ).to(device)

        q, k, v = make_qkv(cfg["batch_size"], seq_len, cfg["d_model"], device)

        out_flash = flash(q, k, v)
        out_local = local(q, k, v)

        assert torch.allclose(out_flash, out_local, atol=1e-4), \
            f"Max diff: {(out_flash - out_local).abs().max()}"

    def test_gradient_flow(self, default_config, device):
        """All backends should support gradient computation."""
        cfg = default_config

        backends = {
            "dense": DenseAttention(cfg["d_model"], cfg["num_heads"], causal=True).to(device),
            "flash": FlashAttention(cfg["d_model"], cfg["num_heads"], causal=True).to(device),
            "local": LocalAttention(cfg["d_model"], cfg["num_heads"], causal=True, window_size=16).to(device),
        }

        for name, backend in backends.items():
            q, k, v = make_qkv(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], device)
            q.requires_grad_(True)
            k.requires_grad_(True)
            v.requires_grad_(True)

            out = backend(q, k, v)
            loss = out.sum()
            loss.backward()

            assert q.grad is not None, f"{name}: Q gradient is None"
            assert k.grad is not None, f"{name}: K gradient is None"
            assert v.grad is not None, f"{name}: V gradient is None"
            assert not torch.isnan(q.grad).any(), f"{name}: Q gradient has NaN"
            assert not torch.isnan(k.grad).any(), f"{name}: K gradient has NaN"
            assert not torch.isnan(v.grad).any(), f"{name}: V gradient has NaN"


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_token(self, device):
        """Test with sequence length of 1."""
        d_model, num_heads = 64, 4

        backends = [
            DenseAttention(d_model, num_heads, causal=True).to(device),
            FlashAttention(d_model, num_heads, causal=True).to(device),
            LocalAttention(d_model, num_heads, causal=True, window_size=8).to(device),
        ]

        q, k, v = make_qkv(2, 1, d_model, device)

        for backend in backends:
            out = backend(q, k, v)
            assert out.shape == (2, 1, d_model)
            assert not torch.isnan(out).any()

    def test_batch_size_one(self, device):
        """Test with batch size of 1."""
        d_model, num_heads, seq_len = 64, 4, 32

        backends = [
            DenseAttention(d_model, num_heads, causal=True).to(device),
            FlashAttention(d_model, num_heads, causal=True).to(device),
            LocalAttention(d_model, num_heads, causal=True, window_size=8).to(device),
        ]

        q, k, v = make_qkv(1, seq_len, d_model, device)

        for backend in backends:
            out = backend(q, k, v)
            assert out.shape == (1, seq_len, d_model)
            assert not torch.isnan(out).any()

    def test_large_batch(self, device):
        """Test with larger batch size."""
        d_model, num_heads, seq_len = 64, 4, 16

        backends = [
            DenseAttention(d_model, num_heads, causal=True).to(device),
            FlashAttention(d_model, num_heads, causal=True).to(device),
            LocalAttention(d_model, num_heads, causal=True, window_size=8).to(device),
        ]

        q, k, v = make_qkv(16, seq_len, d_model, device)

        for backend in backends:
            out = backend(q, k, v)
            assert out.shape == (16, seq_len, d_model)
            assert not torch.isnan(out).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

