import torch
import time

# Initialize the matrix
matrix = torch.randn(1024, 128000, device='cuda', dtype=torch.float32)

# Warm up
_ = torch.nn.functional.softmax(matrix, dim=-1)

total_time = 0
n_iters = 5

for i in range(n_iters):
    # Measure time
    torch.cuda.synchronize()  # Ensure all CUDA operations are finished
    start = time.time()
    _ = torch.nn.functional.softmax(matrix, dim=-1)
    torch.cuda.synchronize()  # Synchronize again
    end = time.time()
    
    total_time += (end - start) * 1000
    print(total_time)

print(f"Softmax computation time (average): {(total_time/n_iters)} ms")
