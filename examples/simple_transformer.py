import torch
from autoattn import AutoAttention

def main():
    d_model = 64
    num_heads = 4
    seq_len = 32
    batch_size = 2

    attn = AutoAttention(d_model=d_model, num_heads=num_heads, causal=True, mode="auto")

    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)

    out = attn(q, k, v)
    print("out shape:", out.shape)

if __name__ == "__main__":
    main()
