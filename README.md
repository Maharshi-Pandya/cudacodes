# CUDA Code Explorations

This repository showcases my journey learning and experimenting with CUDA, primarily using C/C++. Inside, you'll find code examples, insights, and optimizations related to various CUDA concepts. Each directory focuses on a specific topic, with extensively commented code to guide understanding.

## What's Inside?

This repo contains:

*   **Practical Examples:** Hands-on CUDA code for common operations.
*   **Optimizations:** Explores different optimization techniques to improve performance.
*   **Well-Commented Code:** Every line is explained, making it easy to follow along.
*   **Learning Resource:**  A place to learn from and improve your own CUDA skills.

## Quick Navigation

*   **`softmax/`**: Explores various softmax implementations, from naive to optimized.
*   **`matmul/`**: Demonstrates matrix multiplication with tiling strategies.
*   **`matvec/`**: Iteratively optimizes matrix-vector multiplication (SGEMV) to achieve cuBLAS-like performance.
*   **`query-device/`**:  A simple tool to get device information of GPU.
*   **`flash-attention/`**: Exploration of flash attention algorithms (Work in progress).

## Getting Started

1.  Clone the repository:
   ```
    git clone https://github.com/Maharshi-Pandya/cudacodes.git
   ```
2. Navigate to the directory of interest (e.g., `cd matmul`)
3. Check the `README.md` inside the directory to learn how to compile and run the examples.
