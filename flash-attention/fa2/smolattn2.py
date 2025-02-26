import os
import time
import torch
import math
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# Set CUDA architecture for RTX 4090
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

# Load CUDA kernel
smolattn = load(name='smolattn', sources=['build.cpp', 'flash-attn-2.cu'], extra_cuda_cflags=['-O3'])

# Model parameters
batch_size = 16
n_head = 8
seq_len = 512
head_embd = 64

q = torch.randn((batch_size, n_head, seq_len, head_embd), device="cuda")
k = torch.randn((batch_size, n_head, seq_len, head_embd), device="cuda")
v = torch.randn((batch_size, n_head, seq_len, head_embd), device="cuda")

# Manual attention function
def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

# Function to measure execution time
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


print(f"Batch size: {batch_size}, Num heads: {n_head}, Sequence length: {seq_len}, Head dims: {head_embd}\n")

# Benchmarking manual attention
print("=== Benchmarking Manual Attention ===")
manual_result, manual_time = benchmark(manual_attn, q, k, v, name="Manual Attention")

# Benchmarking smolattn implementation
print("\n=== Benchmarking SmolAttn Implementation ===")
minimal_result, smolattn_time = benchmark(smolattn.fa2_forward, q, k, v, name="SmolAttn2")

# Print speedup
speedup = manual_time / smolattn_time
print(f"\nSmolAttn2 is {speedup:.2f}x faster than Manual Attention.")

# Sanity check for correctness
print("\n=== Accuracy Check ===")
tolerance = 1e-2
allclose = torch.allclose(minimal_result, manual_result, rtol=0, atol=tolerance)
print(f"Attn values match within tolerance ({tolerance}): {allclose}")

# Compute absolute differences
diff = torch.abs(minimal_result - manual_result)
diff_indices = torch.nonzero(diff > tolerance, as_tuple=True)

# Print the top mismatches
if diff_indices[0].numel() > 0:
    print("\nTop mismatches:")
    for idx in zip(*diff_indices[:4]):  # Print first 4 mismatches
        print(f"Index {idx}: manual={manual_result[idx].item():.6f}, minimal={minimal_result[idx].item():.6f}, diff={diff[idx].item():.6f}")
else:
    print("\nNo significant mismatches found.")
