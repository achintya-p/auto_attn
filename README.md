# autoattn

Automatic routing between attention backends for LLMs/VLMs.

## Install

```bash
pip install -e .
```

## Usage

```python
import torch
from autoattn import AutoAttention

attn = AutoAttention(d_model=256, num_heads=8, causal=True)

q = torch.randn(2, 128, 256)
k = torch.randn(2, 128, 256)
v = torch.randn(2, 128, 256)

out = attn(q, k, v)  # Automatically picks best backend
```

## Backends

| Backend | When Used | Memory | Exact? |
|---------|-----------|--------|--------|
| `dense` | CPU, fallback | O(N²) | ✅ |
| `flash` | GPU, seq ≤ 2048 | O(N) | ✅ |
| `local` | GPU, seq > 4096, memory mode | O(N·W) | ❌ |

## Modes

```python
# Auto (default) - picks based on device/seq length
AutoAttention(d_model=256, num_heads=8, mode="auto")

# Performance - prefer flash on GPU
AutoAttention(d_model=256, num_heads=8, mode="performance")

# Memory - prefer local/sparse
AutoAttention(d_model=256, num_heads=8, mode="memory")
```

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0

