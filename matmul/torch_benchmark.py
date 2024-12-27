# simple benchmark

import torch
import time

matrix_size = 4096  # (1024 x 1024)
max_val = 10
min_val = -10

# Initialize random tensors with a normal distribution and clamp values
A = torch.randn((matrix_size, matrix_size)).clamp(min=min_val, max=max_val)
B = torch.randn((matrix_size, matrix_size)).clamp(min=min_val, max=max_val)

A, B = A.cuda(), B.cuda()

print(f">> Benchmarking torch.matmul for {matrix_size} x {matrix_size} matrices...")
start_time = time.time()
C = torch.matmul(A, B)
end_time = time.time()

elapsed_time_ms = (end_time - start_time) * 1000

print(f">> Matrix multiplication completed in {elapsed_time_ms:.3f} ms.")