import matplotlib.pyplot as plt
import numpy as np
plt.style.use('fivethirtyeight')

matrix_sizes = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
custom_cuda_time = [0.739136, 0.732832, 1.107200, 2.074880, 4.109728, 8.162368, 16.282017]
torch_time = [3.1140804290771484, 3.124523162841797, 3.125, 3.1251907348632812, 
             6.25004768371582, 12.506628036499023, 25.00143051147461]

plt.figure(figsize=(12, 7))
plt.plot(matrix_sizes, custom_cuda_time, label='Custom CUDA softmax', 
        marker='o', linestyle='-', linewidth=2, color='#2ecc71')
plt.plot(matrix_sizes, torch_time, label='PyTorch softmax',
        marker='s', linestyle='--', linewidth=2, color='#e74c3c')

plt.title('Softmax: Execution Time Comparison', fontsize=16, pad=20)
plt.xlabel('Matrix Column Size (N)', fontsize=14)
plt.ylabel('Execution Time (ms)', fontsize=14)
plt.xticks(matrix_sizes, rotation=90, fontsize=10)
plt.yticks(fontsize=12)

plt.grid(True, alpha=0.9)
plt.legend(fontsize=12, framealpha=0.8)
plt.tight_layout()
plt.show()