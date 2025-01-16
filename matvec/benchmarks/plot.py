import matplotlib.pyplot as plt
import numpy as np

# Data
matrix_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]

custom_cuda_gflops = [1.018906, 19.980488, 26.554296, 40.169170, 47.766766, 49.741516, 50.872116]
cublas_gflops = [1.280000, 17.102297, 32.347481, 46.022472, 49.813587, 44.747833, 50.334869]

custom_cuda_bandwidth = [1.846255, 35.926109, 47.561153, 71.806648, 85.304932, 88.788231, 90.784180]
cublas_bandwidth = [2.319358, 30.750952, 57.937275, 82.270042, 88.960274, 79.874550, 89.825439]

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot GFLOPS
axes[0].plot(matrix_sizes, custom_cuda_gflops, label='Custom CUDA Kernel', marker='o', linestyle='-', linewidth=2)
axes[0].plot(matrix_sizes, cublas_gflops, label='cuBLAS', marker='s', linestyle='--', linewidth=2)
axes[0].set_title('SGEMV: GFLOPS Comparison', fontsize=16)
axes[0].set_xlabel('Matrix Size (M)', fontsize=14)
axes[0].set_ylabel('Achieved GFLOPS', fontsize=14)
axes[0].set_xticks(matrix_sizes)
axes[0].tick_params(axis='x', labelrotation=90, labelsize=10)
axes[0].tick_params(axis='y', labelsize=12)
axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)
axes[0].legend(fontsize=12)

# Plot Bandwidth
axes[1].plot(matrix_sizes, custom_cuda_bandwidth, label='Custom CUDA Kernel', marker='o', linestyle='-', linewidth=2)
axes[1].plot(matrix_sizes, cublas_bandwidth, label='cuBLAS', marker='s', linestyle='--', linewidth=2)
axes[1].set_title('SGEMV: Memory Bandwidth Comparison', fontsize=16)
axes[1].set_xlabel('Matrix Size (M)', fontsize=14)
axes[1].set_ylabel('Achieved Memory Bandwidth (%)', fontsize=14)
axes[1].set_xticks(matrix_sizes)
axes[1].tick_params(axis='x', labelrotation=90, labelsize=10)
axes[1].tick_params(axis='y', labelsize=12)
axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)
axes[1].legend(fontsize=12)

# Adjust layout and show plot
plt.tight_layout()
plt.show()