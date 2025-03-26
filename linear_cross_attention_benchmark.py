import torch
import torch.nn as nn

from linear_cross_attention_experiment import LinearCrossAttention

class SoftmaxCrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, q, k, v):
        Q = self.query_proj(q)
        K = self.key_proj(k)
        V = self.value_proj(v)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, Lq, Lk)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, V)


import time
import numpy as np

def benchmark_attention(attn_class, seq_lens, dim=64, batch_size=8, repeat=10):
    times = []

    for L in seq_lens:
        attn = attn_class(dim)
        q = torch.randn(batch_size, L, dim)
        k = torch.randn(batch_size, L, dim)
        v = torch.randn(batch_size, L, dim)

        for _ in range(3):
            _ = attn(q, k, v)

        torch.cuda.empty_cache()
        start = time.time()
        for _ in range(repeat):
            _ = attn(q, k, v)
        end = time.time()

        avg_time = (end - start) / repeat
        times.append(avg_time)

    return np.array(times)

seq_lens = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

# Запуск
print("Benchmarking LinearCrossAttention...")
time_linear = benchmark_attention(LinearCrossAttention, seq_lens)

print("Benchmarking SoftmaxCrossAttention...")
time_softmax = benchmark_attention(SoftmaxCrossAttention, seq_lens)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(seq_lens, time_softmax * 1000, label='Softmax Attention')
plt.plot(seq_lens, time_linear * 1000, label='Linear Attention')
plt.xlabel("Sequence Length")
plt.ylabel("Time per forward pass (ms)")
plt.title("Attention Time Comparison (batch=8, dim=64)")
plt.legend()
plt.grid(True)
plt.show()

