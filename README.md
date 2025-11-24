# dgx_spark_flash-attn_evo2
The complied whl of flash-attn_2.8.0.post2  for dgx_spark to install evo2, please see the release button of the right side in the repo

## How it was compiled

The dependent envrionment can be seen in this repo, the yaml, and pip list files.

A tricky install is the transformer_engine_torch for evo2, you can solve it by 
```
CUDA_HOME=$CONDA_PREFIX CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-build-isolation --no-cache-dir transformer_engine_torch==2.8.0
```

After install the require environment, you can complied from source by the following command
```shell
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout -b 2.8.0.post2
export NVCC_THREADS=10
nohup python setup.py bdist_wheel > build_log_20251124.txt 2>&1 &
```

The dissussion of the  compling process can be seen in [https://github.com/Dao-AILab/flash-attention/issues/1969](https://github.com/Dao-AILab/flash-attention/issues/1969#issuecomment-3539620956), please remember, it would have error when compiling with ninja, you have better build in just one thread

Finnaly you can install evo2 by 
```shell
pip install evo2
```


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

### Usage of flash-attention
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

## Usage of evo2

The quick test is run the following code

```shell
python -m evo2.test.test_evo2_generation --model_name evo2_7b
```

Here is the output

```markdown
[11/24/25 10:43:11] INFO     httpx - INFO - HTTP Request: GET   _client.py:1025
                             https://hf-mirror.com/api/models/a                
                             rcinstitute/evo2_7b/revision/main                 
                             "HTTP/1.1 200 OK"                                 
Fetching 4 files: 100%|███████████████████████| 4/4 [00:00<00:00, 57260.12it/s]
Found complete file in repo: evo2_7b.pt
                    INFO     StripedHyena - INFO - Initializing    model.py:616
                             StripedHyena with config:                         
                             {'model_name':                                    
                             'shc-evo2-7b-8k-2T-v2', 'vocab_size':             
                             512, 'hidden_size': 4096,                         
                             'num_filters': 4096,                              
                             'hcl_layer_idxs': [2, 6, 9, 13, 16,               
                             20, 23, 27, 30], 'hcm_layer_idxs':                
                             [1, 5, 8, 12, 15, 19, 22, 26, 29],                
                             'hcs_layer_idxs': [0, 4, 7, 11, 14,               
                             18, 21, 25, 28], 'attn_layer_idxs':               
                             [3, 10, 17, 24, 31],                              
                             'hcm_filter_length': 128,                         
                             'hcl_filter_groups': 4096,                        
                             'hcm_filter_groups': 256,                         
                             'hcs_filter_groups': 256,                         
                             'hcs_filter_length': 7, 'num_layers':             
                             32, 'short_filter_length': 3,                     
                             'num_attention_heads': 32,                        
                             'short_filter_bias': False,                       
                             'mlp_init_method':                                
                             'torch.nn.init.zeros_',                           
                             'mlp_output_init_method':                         
                             'torch.nn.init.zeros_', 'eps': 1e-06,             
                             'state_size': 16, 'rotary_emb_base':              
                             10000, 'rotary_emb_scaling_factor':               
                             128,                                              
                             'use_interpolated_rotary_pos_emb':                
                             True, 'make_vocab_size_divisible_by':             
                             8, 'inner_size_multiple_of': 16,                  
                             'inner_mlp_size': 11264,                          
                             'log_intermediate_values': False,                 
                             'proj_groups': 1,                                 
                             'hyena_filter_groups': 1,                         
                             'column_split_hyena': False,                      
                             'column_split': True, 'interleave':               
                             True, 'evo2_style_activations': True,             
                             'model_parallel_size': 1,                         
                             'pipe_parallel_size': 1,                          
                             'tie_embeddings': True,                           
                             'mha_out_proj_bias': True,                        
                             'hyena_out_proj_bias': True,                      
                             'hyena_flip_x1x2': False,                         
                             'qkv_proj_bias': False,                           
                             'use_fp8_input_projections': True,                
                             'max_seqlen': 1048576,                            
                             'max_batch_size': 1, 'final_norm':                
                             True, 'use_flash_attn': True,                     
                             'use_flash_rmsnorm': False,                       
                             'use_flash_depthwise': False,                     
                             'use_flashfft': False,                            
                             'use_laughing_hyena': False,                      
                             'inference_mode': True,                           
                             'tokenizer_type':                                 
                             'CharLevelTokenizer',                             
                             'prefill_style': 'fft',                           
                             'mlp_activation': 'gelu',                         
                             'print_activations': False, 'Loader':             
                             <class 'yaml.loader.FullLoader'>}                 
                    INFO     StripedHyena - INFO - Initializing 32 model.py:635
                             blocks...                                         
                    INFO     StripedHyena - INFO - Distributing    model.py:642
                             across 1 GPUs, approximately 32                   
                             layers per GPU                                    
  0%|                                                   | 0/32 [00:00<?, ?it/s]                    INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=0 to device='cuda:0'                    
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 0: 205571840                            
                    INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=1 to device='cuda:0'                    
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 1: 205606912                            
                    INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=2 to device='cuda:0'                    
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 2: 205705216                            
  9%|████                                       | 3/32 [00:00<00:01, 25.00it/s][11/24/25 10:43:12] INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=3 to device='cuda:0'                    
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 3: 205533184                            
                    INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=4 to device='cuda:0'                    
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 4: 205571840                            
                    INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=5 to device='cuda:0'                    
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 5: 205606912                            
 19%|████████                                   | 6/32 [00:00<00:01, 18.72it/s]                    INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=6 to device='cuda:0'                    
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 6: 205705216                            
                    INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=7 to device='cuda:0'                    
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 7: 205571840                            
                    INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=8 to device='cuda:0'                    
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 8: 205606912                            
                    INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=9 to device='cuda:0'                    
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 9: 205705216                            
                    INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=10 to device='cuda:0'                   
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 10: 205533184                           
 34%|██████████████▍                           | 11/32 [00:00<00:00, 28.59it/s]                    INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=11 to device='cuda:0'                   
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 11: 205571840                           
                    INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=12 to device='cuda:0'                   
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 12: 205606912                           
                    INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=13 to device='cuda:0'                   
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 13: 205705216                           
                    INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=14 to device='cuda:0'                   
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 14: 205571840                           
                    INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=15 to device='cuda:0'                   
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 15: 205606912                           
 50%|█████████████████████                     | 16/32 [00:00<00:00, 34.76it/s]                    INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=16 to device='cuda:0'                   
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 16: 205705216                           
                    INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=17 to device='cuda:0'                   
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 17: 205533184                           
                    INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=18 to device='cuda:0'                   
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 18: 205571840                           
                    INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=19 to device='cuda:0'                   
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 19: 205606912                           
                    INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=20 to device='cuda:0'                   
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 20: 205705216                           
 66%|███████████████████████████▌              | 21/32 [00:00<00:00, 38.36it/s]                    INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=21 to device='cuda:0'                   
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 21: 205571840                           
                    INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=22 to device='cuda:0'                   
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 22: 205606912                           
                    INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=23 to device='cuda:0'                   
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 23: 205705216                           
                    INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=24 to device='cuda:0'                   
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 24: 205533184                           
                    INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=25 to device='cuda:0'                   
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 25: 205571840                           
 81%|██████████████████████████████████▏       | 26/32 [00:00<00:00, 40.81it/s]                    INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=26 to device='cuda:0'                   
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 26: 205606912                           
                    INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=27 to device='cuda:0'                   
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 27: 205705216                           
                    INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=28 to device='cuda:0'                   
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 28: 205571840                           
                    INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=29 to device='cuda:0'                   
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 29: 205606912                           
                    INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=30 to device='cuda:0'                   
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 30: 205705216                           
 97%|████████████████████████████████████████▋ | 31/32 [00:00<00:00, 42.71it/s]                    INFO     StripedHyena - INFO - Assigned        model.py:660
                             layer_idx=31 to device='cuda:0'                   
                    INFO     StripedHyena - INFO - Parameter count model.py:661
                             for block 31: 205533184                           
100%|██████████████████████████████████████████| 32/32 [00:00<00:00, 36.68it/s]
                    INFO     StripedHyena - INFO - Initialized     model.py:680
                             model                                             
                    INFO     vortex.model.utils - INFO - Loading    utils.py:92
                             /home/wlt2025/.cache/huggingface/hub/m            
                             odels--arcinstitute--evo2_7b/snapshots            
                             /bb22e3e420fcdac33aff6de942bf0911c9a32            
                             22d/evo2_7b.pt                                    
Extra keys in state_dict: {'blocks.10.mixer.dense._extra_state', 'blocks.23.mixer.mixer.filter.t', 'blocks.3.mixer.attn._extra_state', 'blocks.24.mixer.attn._extra_state', 'blocks.27.mixer.mixer.filter.t', 'blocks.13.mixer.mixer.filter.t', 'blocks.3.mixer.dense._extra_state', 'blocks.30.mixer.mixer.filter.t', 'blocks.17.mixer.attn._extra_state', 'blocks.2.mixer.mixer.filter.t', 'blocks.31.mixer.dense._extra_state', 'blocks.31.mixer.attn._extra_state', 'blocks.10.mixer.attn._extra_state', 'blocks.20.mixer.mixer.filter.t', 'unembed.weight', 'blocks.17.mixer.dense._extra_state', 'blocks.24.mixer.dense._extra_state', 'blocks.16.mixer.mixer.filter.t', 'blocks.9.mixer.mixer.filter.t', 'blocks.6.mixer.mixer.filter.t'}

..............

GCGAGCAGTAGCCCAAACAATCTCATATGAAGTCACCCTAGCCATCATTCTACTATCAACATTACTAATAAGTGGCTCCTTTAACCTCTCCACCCTTATCACAA",     Output: "CACAAGAACACCTATGACTCCTCCTACCATCATGACCCCTAGCCATAATATGATTTACCTCCACACTAGCAGAAACCAACCGAGCCCCCTTCGACCTAACAGAAGGCGAATCAGAACTAGTCTCAGGCTTCAACATCGAATACGCCGCAGGCTCATTCGCCCTATTCTTCATAGCAGAATACATAAACATCATCATAATAAACGCCCTAACCACCACCATCTTCCTAGCCACACCACACAACCTAACCACACCAGAACTCTACACAACAAACTTCACCACCAAAACCCTCCTCCTAACCACCCTATTCCTATGAATCCGAGCAACCTACCCCCGATTCCGCTACGACCAACTCATACACCTACTATGAAAAAACTTCCTACCCCTCACACTAGCACTATGCATATGATACGTCTCAATACCCATCCTACTATCCGGCATCCCCCCACAAACATAAGAAATATGTCTGACAAAAGAGTTACTTTGATAGAGTAAATAATAG",   Score: -0.08920665085315704

Test Results:
% Matching Nucleotides: 89.35

Test Passed! Score matches expected 89.25%

```













