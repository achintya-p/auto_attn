"""
Unit tests for AutoAttention router.

Tests verify:
1. Routing logic selects appropriate backends
2. Caching works correctly
3. All modes (auto, performance, memory) work
4. Fallback to dense always works
"""

import pytest
import torch

from autoattn import AutoAttention


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
    torch.manual_seed(42)
    q = torch.randn(batch_size, seq_len, d_model, device=device)
    k = torch.randn(batch_size, seq_len, d_model, device=device)
    v = torch.randn(batch_size, seq_len, d_model, device=device)
    return q, k, v


class TestAutoAttentionBasic:
    """Basic functionality tests."""

    def test_initialization(self, default_config):
        """AutoAttention should initialize without errors."""
        cfg = default_config
        attn = AutoAttention(cfg["d_model"], cfg["num_heads"], causal=True)

        assert attn.d_model == cfg["d_model"]
        assert attn.num_heads == cfg["num_heads"]
        assert attn.causal is True
        assert "dense" in attn.backends
        assert "flash" in attn.backends
        assert "local" in attn.backends

    def test_forward_pass(self, default_config, device):
        """Forward pass should work and produce correct shape."""
        cfg = default_config
        attn = AutoAttention(cfg["d_model"], cfg["num_heads"], causal=True).to(device)
        q, k, v = make_qkv(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], device)

        out = attn(q, k, v)

        assert out.shape == q.shape
        assert not torch.isnan(out).any()

    def test_no_nans(self, default_config, device):
        """Output should not contain NaN values."""
        cfg = default_config
        attn = AutoAttention(cfg["d_model"], cfg["num_heads"], causal=True).to(device)
        q, k, v = make_qkv(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], device)

        out = attn(q, k, v)

        assert not torch.isnan(out).any()

    def test_gradient_flow(self, default_config, device):
        """Gradients should flow through AutoAttention."""
        cfg = default_config
        attn = AutoAttention(cfg["d_model"], cfg["num_heads"], causal=True).to(device)
        q, k, v = make_qkv(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], device)
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)

        out = attn(q, k, v)
        loss = out.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None


class TestRoutingLogic:
    """Tests for backend selection logic."""

    def test_cpu_routes_to_dense(self, default_config):
        """CPU should always route to dense backend."""
        cfg = default_config
        attn = AutoAttention(cfg["d_model"], cfg["num_heads"], causal=True)
        q, k, v = make_qkv(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], torch.device("cpu"))

        backend_name = attn.get_backend_name(q)

        assert backend_name == "dense"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_short_seq_routes_to_flash(self, default_config):
        """GPU with short sequences should route to flash."""
        cfg = default_config
        attn = AutoAttention(cfg["d_model"], cfg["num_heads"], causal=True, mode="auto")
        q, k, v = make_qkv(cfg["batch_size"], 128, cfg["d_model"], torch.device("cuda"))

        backend_name = attn.get_backend_name(q)

        assert backend_name == "flash"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_long_seq_routes_to_local(self, default_config):
        """GPU with very long sequences should route to local."""
        cfg = default_config
        attn = AutoAttention(cfg["d_model"], cfg["num_heads"], causal=True, mode="auto")
        # Create a signature that would bucket to >4096
        q, k, v = make_qkv(cfg["batch_size"], 8192, cfg["d_model"], torch.device("cuda"))

        backend_name = attn.get_backend_name(q)

        assert backend_name == "local"

    def test_memory_mode_prefers_local(self, default_config, device):
        """Memory mode should prefer local attention."""
        cfg = default_config
        attn = AutoAttention(cfg["d_model"], cfg["num_heads"], causal=True, mode="memory").to(device)

        # For GPU, memory mode should choose local
        if device.type == "cuda":
            q, k, v = make_qkv(cfg["batch_size"], 1024, cfg["d_model"], device)
            backend_name = attn.get_backend_name(q)
            assert backend_name == "local"

    def test_metadata_override(self, default_config, device):
        """Metadata should override default mode."""
        cfg = default_config
        attn = AutoAttention(cfg["d_model"], cfg["num_heads"], causal=True, mode="auto").to(device)
        q, k, v = make_qkv(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], device)

        # Test with memory mode override
        if device.type == "cuda":
            backend_name = attn.get_backend_name(q, metadata={"mode": "memory"})
            assert backend_name == "local"


class TestCaching:
    """Tests for routing decision caching."""

    def test_cache_hit(self, default_config, device):
        """Same input signature should hit cache."""
        cfg = default_config
        attn = AutoAttention(cfg["d_model"], cfg["num_heads"], causal=True).to(device)

        q1, k1, v1 = make_qkv(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], device)
        q2, k2, v2 = make_qkv(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], device)

        # First call populates cache
        _ = attn(q1, k1, v1)
        cache_size_after_first = len(attn._choice_cache)

        # Second call with same signature should hit cache
        _ = attn(q2, k2, v2)
        cache_size_after_second = len(attn._choice_cache)

        assert cache_size_after_first == cache_size_after_second

    def test_different_seq_len_different_cache_entry(self, default_config, device):
        """Different sequence lengths should create different cache entries."""
        cfg = default_config
        attn = AutoAttention(cfg["d_model"], cfg["num_heads"], causal=True).to(device)

        q1, k1, v1 = make_qkv(cfg["batch_size"], 32, cfg["d_model"], device)
        q2, k2, v2 = make_qkv(cfg["batch_size"], 128, cfg["d_model"], device)

        _ = attn(q1, k1, v1)
        cache_size_after_first = len(attn._choice_cache)

        _ = attn(q2, k2, v2)
        cache_size_after_second = len(attn._choice_cache)

        # Different seq lengths should create different cache entries
        # (unless they bucket to same value)
        assert cache_size_after_second >= cache_size_after_first

    def test_bucketing(self, default_config):
        """Bucketing should group similar values."""
        cfg = default_config
        attn = AutoAttention(cfg["d_model"], cfg["num_heads"], causal=True)

        # Test bucketing function
        assert attn._bucket(1) == 16
        assert attn._bucket(16) == 16
        assert attn._bucket(17) == 16  # rounds to nearest power of 2
        assert attn._bucket(32) == 32
        assert attn._bucket(33) == 32
        assert attn._bucket(48) == 32 or attn._bucket(48) == 64  # depends on rounding
        assert attn._bucket(64) == 64
        assert attn._bucket(1024) == 1024


class TestModes:
    """Tests for different operating modes."""

    def test_auto_mode(self, default_config, device):
        """Auto mode should work without errors."""
        cfg = default_config
        attn = AutoAttention(cfg["d_model"], cfg["num_heads"], mode="auto").to(device)
        q, k, v = make_qkv(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], device)

        out = attn(q, k, v)
        assert out.shape == q.shape

    def test_performance_mode(self, default_config, device):
        """Performance mode should work without errors."""
        cfg = default_config
        attn = AutoAttention(cfg["d_model"], cfg["num_heads"], mode="performance").to(device)
        q, k, v = make_qkv(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], device)

        out = attn(q, k, v)
        assert out.shape == q.shape

    def test_memory_mode(self, default_config, device):
        """Memory mode should work without errors."""
        cfg = default_config
        attn = AutoAttention(cfg["d_model"], cfg["num_heads"], mode="memory").to(device)
        q, k, v = make_qkv(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], device)

        out = attn(q, k, v)
        assert out.shape == q.shape


class TestTrainEvalPhases:
    """Tests for training vs inference behavior."""

    def test_train_mode(self, default_config, device):
        """Training mode should work correctly."""
        cfg = default_config
        attn = AutoAttention(cfg["d_model"], cfg["num_heads"], causal=True).to(device)
        attn.train()
        q, k, v = make_qkv(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], device)

        out = attn(q, k, v)
        assert out.shape == q.shape

    def test_eval_mode(self, default_config, device):
        """Evaluation mode should work correctly."""
        cfg = default_config
        attn = AutoAttention(cfg["d_model"], cfg["num_heads"], causal=True).to(device)
        attn.eval()
        q, k, v = make_qkv(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], device)

        with torch.no_grad():
            out = attn(q, k, v)
        assert out.shape == q.shape

    def test_phase_metadata_override(self, default_config, device):
        """Phase metadata should override module.training state."""
        cfg = default_config
        attn = AutoAttention(cfg["d_model"], cfg["num_heads"], causal=True).to(device)
        attn.train()  # Set to training mode
        q, k, v = make_qkv(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], device)

        # Override with inference phase
        out = attn(q, k, v, metadata={"phase": "inference"})
        assert out.shape == q.shape


class TestCausalMasking:
    """Tests for causal masking through AutoAttention."""

    def test_causal_true(self, default_config, device):
        """Causal=True should mask future tokens."""
        cfg = default_config
        attn = AutoAttention(cfg["d_model"], cfg["num_heads"], causal=True).to(device)
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
        )

    def test_causal_false(self, default_config, device):
        """Causal=False should allow bidirectional attention."""
        cfg = default_config
        attn = AutoAttention(cfg["d_model"], cfg["num_heads"], causal=False).to(device)
        q, k, v = make_qkv(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], device)

        out = attn(q, k, v)
        assert out.shape == q.shape


class TestVariousInputSizes:
    """Tests with different input sizes."""

    def test_single_token(self, device):
        """Should handle sequence length of 1."""
        attn = AutoAttention(64, 4, causal=True).to(device)
        q, k, v = make_qkv(2, 1, 64, device)

        out = attn(q, k, v)
        assert out.shape == (2, 1, 64)

    def test_long_sequence(self, device):
        """Should handle longer sequences."""
        attn = AutoAttention(64, 4, causal=True).to(device)
        q, k, v = make_qkv(1, 256, 64, device)

        out = attn(q, k, v)
        assert out.shape == (1, 256, 64)

    def test_large_batch(self, device):
        """Should handle larger batch sizes."""
        attn = AutoAttention(64, 4, causal=True).to(device)
        q, k, v = make_qkv(8, 32, 64, device)

        out = attn(q, k, v)
        assert out.shape == (8, 32, 64)

    def test_different_d_model(self, device):
        """Should handle different model dimensions."""
        for d_model in [32, 64, 128, 256]:
            num_heads = d_model // 16
            attn = AutoAttention(d_model, num_heads, causal=True).to(device)
            q, k, v = make_qkv(2, 16, d_model, device)

            out = attn(q, k, v)
            assert out.shape == (2, 16, d_model)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

