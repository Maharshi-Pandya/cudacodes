import torch
import time
import numpy as np
import os


def random_normal_clamped(min_val, max_val, size):
    return np.clip(np.random.normal(0, 1, size), min_val, max_val)


def benchmark_kernel_for_sizes(min_n, max_n):
    os.makedirs('benchmarks', exist_ok=True)
    n_iters = 5
    
    with open('benchmarks/exec_time_ms_torch.txt', 'w') as exec_time_file:
        N = min_n
        while N < max_n:
            M = 1024  # matrix size (M, N)
            print(f'------------ Running PyTorch softmax benchmark for MxN = ({M}, {N}) -------------')

            # Generate random data
            mat = torch.tensor(random_normal_clamped(-10, 10, (M, N)), dtype=torch.float32, device="cuda")
            torch.cuda.synchronize()
            
            # Warmup
            for _ in range(5):
                _ = torch.nn.functional.softmax(mat, dim=-1)
                torch.cuda.synchronize()
            
            total_time = 0
            # Run softmax kernel
            for i in range(n_iters):
                # Measure time
                torch.cuda.synchronize()  # Ensure all CUDA operations are finished
                start = time.time()
                _ = torch.nn.functional.softmax(mat, dim=-1)
                torch.cuda.synchronize()  # Synchronize again
                end = time.time()
                
                total_time += (end - start) * 1000

            exec_time = (total_time/n_iters)
            print(f'>> Kernel execution time: {exec_time:.3f} ms')

            # Write execution time to file
            exec_time_file.write(f'{M} {exec_time}\n')

            # Clear GPU memory
            del mat
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
                        
            N *= 2
            time.sleep(1)


if __name__ == '__main__':
    benchmark_kernel_for_sizes(2048, 262144)
