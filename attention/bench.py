import os
import time
import math
import torch
torch.backends.cuda.matmul.allow_tf32 = True
from torch.nn import functional as F
from torch.utils.cpp_extension import load
from pycublas import cublas_create_handle, cublas_destroy_handle

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# on H100
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
plt.style.use('Solarize_Light2')

handle = cublas_create_handle()
smolattn = load(name='smolattn', sources=['build.cpp', 'attn.cu'],
                extra_cuda_cflags=['-O3', '-arch=sm_89', '-lcublas'])

# Fixed model parameters
batch_size = 2
n_head = 16
head_embd = 64

# Manual attention function
def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

# Benchmarking function to measure execution time
def benchmark(func, *args, name="Function"):
    torch.cuda.synchronize()  # Ensure GPU is idle
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    result = func(*args)
    end.record()
    
    torch.cuda.synchronize()  # Wait for event completion
    elapsed_time = start.elapsed_time(end)  # Time in ms
    print(f"{name} execution time: {elapsed_time:.3f} ms")
    return result, elapsed_time

# List of sequence lengths to evaluate
seq_lens = [512, 1024, 2048, 4096, 8192]

# Lists to store execution times for each implementation
manual_times = []
smolattn_times = []

# Number of warmup iterations
warmup_iters = 3

print("=== Benchmarking across different sequence lengths ===")
for seq_len in seq_lens:
    print(f"\n--- Sequence length: {seq_len} ---")
    # Create random Q, K, V tensors for the current sequence length
    q = torch.randn((batch_size, n_head, seq_len, head_embd), device="cuda")
    k = torch.randn((batch_size, n_head, seq_len, head_embd), device="cuda")
    v = torch.randn((batch_size, n_head, seq_len, head_embd), device="cuda")
    
    # Warm-up for manual attention
    for _ in range(warmup_iters):
        _ = manual_attn(q, k, v)
    # Warm-up for smolattn attention
    for _ in range(warmup_iters):
        _ = smolattn.attention_forward(handle.value, q, k, v)
    torch.cuda.synchronize()
    
    # Benchmark manual attention
    _, m_time = benchmark(manual_attn, q, k, v, name="Manual Attention")
    manual_times.append(m_time)
    
    # Benchmark smolattn implementation
    minimal_result, s_time = benchmark(smolattn.attention_forward, handle.value, q, k, v, name="SmolAttn")
    smolattn_times.append(s_time)
    
    # Sanity check for correctness (optional)
    tolerance = 1e-2
    allclose = torch.allclose(minimal_result, manual_attn(q, k, v), rtol=0, atol=tolerance)
    print(f"Attention values match within tolerance ({tolerance}): {allclose}")

# Cleanup the cuBLAS handle
cublas_destroy_handle(handle)

# Create a DataFrame for plotting
data = {
    "Sequence Length": seq_lens * 2,
    "Time (ms)": manual_times + smolattn_times,
    "Implementation": ["Manual"] * len(seq_lens) + ["SmolAttn"] * len(seq_lens)
}
df = pd.DataFrame(data)
df.to_csv("benchmark.csv")

plt.figure(figsize=(12, 8))

# # Plot the manual times
# plt.plot(seq_lens, manual_times, marker='o', label="Manual", color="#FDB813")  # a golden tone

# # Plot the smolattn times
# plt.plot(seq_lens, smolattn_times, marker='o', label="SmolAttn", color="#D95F02")  # a contrasting color

# Create positions for grouped bars
x = np.arange(len(seq_lens))
width = 0.35

# Plot the bars for Manual and SmolAttn times
plt.bar(x - width/2, smolattn_times, width, label="SmolAttn", color="#D95F02")
plt.bar(x + width/2, manual_times, width, label="Manual", color="#FDB813")


plt.title("Execution Speed: Manual vs SmolAttn")
plt.xlabel("Sequence Length")
plt.ylabel("Execution Time (ms)")
plt.xticks(x, seq_lens)
plt.legend(title="Implementation")
plt.tight_layout()

# Save the figure to a file instead of showing it
plt.savefig("execution_speed_comparison.png")
print("Figure saved as execution_speed_comparison.png")
