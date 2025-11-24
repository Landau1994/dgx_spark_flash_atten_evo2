# dgx_spark_flash-attn_evo2
The complied whl of flash-attn_2.8.0.post2  for dgx_spark to install evo2, please see the release button of the right side in the repo

## How it was compiled

The dependent envrionment can be seen in this repo, the yaml, and pip list files.

After install the require environment, you can complied from source by the following command
```shell
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout -b 2.8.0.post2
export NVCC_THREADS=10
nohup python setup.py bdist_wheel > build_log_20251124.txt 2>&1 &
```

The dissussion of the  compling process can be seen in [https://github.com/Dao-AILab/flash-attention/issues/1969](https://github.com/Dao-AILab/flash-attention/issues/1969#issuecomment-3539620956), please remember, it would have error when compiling with ninja, you have better build in just one thread

## How to use flahs attention, here is the demo code

### Overview
- **Device**: CUDA GPU , dgx-spark
- **Data Type**: bfloat16 (reduced precision for efficiency)
- **Test Configuration**: Batch=2, Sequence=5090, Heads=8, Head_dim=64

### Requirements
- PyTorch 2.9+cu18
- flash-attn library (install via `pip install flash-attn`)
- CUDA-compatible GPU

### Benchmark Results
The benchmark measures:
1. **Execution time** for both implementations
2. **Numerical accuracy** (maximum difference between outputs)
3. **Performance speedup** (FlashAttention2 vs PyTorch SDPA)

### Usage
Simply run the following code to compare performance. Results will show timing and accuracy metrics.


```python
import torch
import time

# Configuration for device and data type
device = "cuda"
dtype = torch.bfloat16

# Import FlashAttention2 function - new API
from flash_attn import flash_attn_func

def make_qkv(B=2, S=5090, H=8, D=64):
    """
    Generate query, key, and value tensors for attention computation.
    
    Args:
        B: Batch size
        S: Sequence length
        H: Number of attention heads
        D: Head dimension
    
    Returns:
        Tuple of (query, key, value) tensors
    """
    q = torch.randn(B, S, H, D, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    return q, k, v

def pytorch_sdpa(q, k, v):
    """
    PyTorch's native scaled dot-product attention implementation.
    
    Args:
        q: Query tensor [B, S, H, D]
        k: Key tensor [B, S, H, D]
        v: Value tensor [B, S, H, D]
    
    Returns:
        Attention output tensor [B, S, H, D]
    """
    # Transpose to [B, H, S, D] for PyTorch's SDPA
    q_, k_, v_ = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
    out = torch.nn.functional.scaled_dot_product_attention(q_, k_, v_)
    # Transpose back to [B, S, H, D]
    return out.transpose(1,2)

def flash_forward(q, k, v, causal=False):
    """
    FlashAttention2 implementation for efficient attention computation.
    
    Args:
        q: Query tensor [B, S, H, D]
        k: Key tensor [B, S, H, D]
        v: Value tensor [B, S, H, D]
        causal: Whether to use causal attention mask
    
    Returns:
        Attention output tensor [B, S, H, D]
    """
    return flash_attn_func(q, k, v, causal=causal)

def benchmark(fn, *args):
    """
    Benchmark function execution time with proper CUDA synchronization.
    
    Args:
        fn: Function to benchmark
        *args: Arguments to pass to the function
    
    Returns:
        Tuple of (execution_time, function_output)
    """
    torch.cuda.synchronize()
    start_time = time.time()
    output = fn(*args)
    torch.cuda.synchronize()
    execution_time = time.time() - start_time
    return execution_time, output

# Main benchmark execution
print("Benchmarking FlashAttention2 vs PyTorch SDPA...\n")

# Generate test tensors
query, key, value = make_qkv()

# Benchmark PyTorch's native SDPA
pytorch_time, pytorch_output = benchmark(pytorch_sdpa, query, key, value)
print(f"PyTorch SDPA time: {pytorch_time*1000:.2f} ms")

# Benchmark FlashAttention2
try:
    flash_time, flash_output = benchmark(flash_forward, query, key, value)
    print(f"FlashAttention2 time: {flash_time*1000:.2f} ms")
    
    # Calculate maximum difference between implementations
    max_difference = (pytorch_output - flash_output).abs().max().item()
    print(f"Maximum difference: {max_difference}")
    
    # Calculate speedup
    speedup = pytorch_time / flash_time
    print(f"FlashAttention2 speedup: {speedup:.2f}x")
    
except Exception as e:
    print(f"FlashAttention2 failed: {e}")
    print("Note: Ensure flash-attn is properly installed for GPU compatibility")

```
Here is the results
```text
Benchmarking FlashAttention2 vs PyTorch SDPA...

PyTorch SDPA time: 2.84 ms
FlashAttention2 time: 1.91 ms
Maximum difference: 0.00048828125
FlashAttention2 speedup: 1.49x
```





