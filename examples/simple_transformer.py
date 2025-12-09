"""
AutoAttention Examples

Demonstrates various use cases of the autoattn library:
1. Basic usage
2. Different operating modes
3. GPU acceleration
4. Routing inspection
5. Integration with a transformer block
"""

import torch
import torch.nn as nn
from autoattn import AutoAttention


def example_basic():
    """Basic usage of AutoAttention."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    d_model = 64
    num_heads = 4
    seq_len = 32
    batch_size = 2

    attn = AutoAttention(d_model=d_model, num_heads=num_heads, causal=True, mode="auto")

    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)

    out = attn(q, k, v)
    print(f"Input shape:  {q.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Backend used: {attn.get_backend_name(q)}")
    print()


def example_modes():
    """Different operating modes: auto, performance, memory."""
    print("=" * 60)
    print("Example 2: Operating Modes")
    print("=" * 60)
    
    d_model = 128
    num_heads = 8
    seq_len = 512
    batch_size = 4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    q = torch.randn(batch_size, seq_len, d_model, device=device)
    k = torch.randn(batch_size, seq_len, d_model, device=device)
    v = torch.randn(batch_size, seq_len, d_model, device=device)
    
    for mode in ["auto", "performance", "memory"]:
        attn = AutoAttention(
            d_model=d_model, 
            num_heads=num_heads, 
            causal=True, 
            mode=mode
        ).to(device)
        
        backend = attn.get_backend_name(q)
        out = attn(q, k, v)
        print(f"Mode: {mode:12} -> Backend: {backend:8} | Output: {out.shape}")
    print()


def example_routing_by_sequence_length():
    """Shows how routing changes with sequence length."""
    print("=" * 60)
    print("Example 3: Routing by Sequence Length")
    print("=" * 60)
    
    d_model = 64
    num_heads = 4
    batch_size = 2
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    attn = AutoAttention(d_model=d_model, num_heads=num_heads, causal=True).to(device)
    
    seq_lengths = [32, 128, 512, 1024, 2048, 4096, 8192]
    
    for seq_len in seq_lengths:
        q = torch.randn(batch_size, seq_len, d_model, device=device)
        backend = attn.get_backend_name(q)
        print(f"seq_len={seq_len:5} -> {backend}")
    print()


def example_gpu_acceleration():
    """GPU usage and performance comparison."""
    print("=" * 60)
    print("Example 4: GPU Acceleration")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU example")
        print()
        return
    
    d_model = 256
    num_heads = 8
    seq_len = 1024
    batch_size = 8
    
    # CPU
    attn_cpu = AutoAttention(d_model=d_model, num_heads=num_heads, causal=True)
    q_cpu = torch.randn(batch_size, seq_len, d_model)
    
    # GPU
    attn_gpu = AutoAttention(d_model=d_model, num_heads=num_heads, causal=True).cuda()
    q_gpu = torch.randn(batch_size, seq_len, d_model, device="cuda")
    k_gpu = torch.randn(batch_size, seq_len, d_model, device="cuda")
    v_gpu = torch.randn(batch_size, seq_len, d_model, device="cuda")
    
    print(f"CPU backend: {attn_cpu.get_backend_name(q_cpu)}")
    print(f"GPU backend: {attn_gpu.get_backend_name(q_gpu)}")
    
    # Warmup
    for _ in range(3):
        _ = attn_gpu(q_gpu, k_gpu, v_gpu)
    torch.cuda.synchronize()
    
    # Benchmark
    import time
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _ = attn_gpu(q_gpu, k_gpu, v_gpu)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    print(f"GPU: 100 forward passes in {elapsed*1000:.2f}ms ({elapsed*10:.3f}ms per pass)")
    print()


def example_training_vs_inference():
    """Training mode vs inference mode."""
    print("=" * 60)
    print("Example 5: Training vs Inference")
    print("=" * 60)
    
    d_model = 64
    num_heads = 4
    seq_len = 32
    batch_size = 2
    
    attn = AutoAttention(d_model=d_model, num_heads=num_heads, causal=True)
    
    q = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    k = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    v = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    
    # Training mode
    attn.train()
    out_train = attn(q, k, v)
    loss = out_train.sum()
    loss.backward()
    
    print(f"Training mode: output shape = {out_train.shape}")
    print(f"  Q gradient shape: {q.grad.shape}")
    print(f"  Gradient norm: {q.grad.norm().item():.4f}")
    
    # Inference mode
    attn.eval()
    with torch.no_grad():
        out_eval = attn(q, k, v)
    
    print(f"Inference mode: output shape = {out_eval.shape}")
    print()


def example_metadata_override():
    """Using metadata to override routing decisions."""
    print("=" * 60)
    print("Example 6: Metadata Override")
    print("=" * 60)
    
    d_model = 64
    num_heads = 4
    seq_len = 128
    batch_size = 2
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    attn = AutoAttention(d_model=d_model, num_heads=num_heads, causal=True, mode="auto").to(device)
    
    q = torch.randn(batch_size, seq_len, d_model, device=device)
    k = torch.randn(batch_size, seq_len, d_model, device=device)
    v = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # Default behavior
    print(f"Default (mode=auto): {attn.get_backend_name(q)}")
    
    # Override with metadata
    print(f"Override to memory:  {attn.get_backend_name(q, metadata={'mode': 'memory'})}")
    print(f"Override to perf:    {attn.get_backend_name(q, metadata={'mode': 'performance'})}")
    print()


class SimpleTransformerBlock(nn.Module):
    """A simple transformer block using AutoAttention."""
    
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # Self-attention with auto-routing
        self.attn = AutoAttention(d_model, num_heads, causal=True, mode="auto")
        self.attn_norm = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )
        self.ff_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attn(x, x, x)
        x = self.attn_norm(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.ff_norm(x + self.dropout(ff_out))
        
        return x


def example_transformer_block():
    """Using AutoAttention in a transformer block."""
    print("=" * 60)
    print("Example 7: Transformer Block Integration")
    print("=" * 60)
    
    d_model = 256
    num_heads = 8
    ff_dim = 1024
    seq_len = 128
    batch_size = 4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create transformer block
    block = SimpleTransformerBlock(d_model, num_heads, ff_dim).to(device)
    
    # Input tensor (e.g., token embeddings)
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # Forward pass
    out = block(x)
    
    print(f"Device: {device}")
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Attention backend: {block.attn.get_backend_name(x)}")
    print(f"Parameters: {sum(p.numel() for p in block.parameters()):,}")
    print()


class SimpleGPT(nn.Module):
    """A minimal GPT-style model using AutoAttention."""
    
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int, 
        num_heads: int, 
        num_layers: int,
        max_seq_len: int = 2048
    ):
        super().__init__()
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(d_model, num_heads, d_model * 4)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits


def example_mini_gpt():
    """A minimal GPT model using AutoAttention."""
    print("=" * 60)
    print("Example 8: Mini GPT Model")
    print("=" * 60)
    
    vocab_size = 1000
    d_model = 128
    num_heads = 4
    num_layers = 4
    seq_len = 64
    batch_size = 2
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SimpleGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers
    ).to(device)
    
    # Random token indices
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Forward pass
    logits = model(tokens)
    
    print(f"Device: {device}")
    print(f"Input tokens: {tokens.shape}")
    print(f"Output logits: {logits.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Show which backends each layer uses
    sample_input = torch.randn(batch_size, seq_len, d_model, device=device)
    backends = [block.attn.get_backend_name(sample_input) for block in model.blocks]
    print(f"Backends per layer: {backends}")
    print()


def main():
    """Run all examples."""
    example_basic()
    example_modes()
    example_routing_by_sequence_length()
    example_gpu_acceleration()
    example_training_vs_inference()
    example_metadata_override()
    example_transformer_block()
    example_mini_gpt()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
