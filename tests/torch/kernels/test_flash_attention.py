import torch
import time
import dlcu
from typing import Tuple, Optional
import torch.nn.functional as F
import math

# Macros to test for correctness
BATCH_SIZE = 1
NUM_HEADS = 128
SEQ_LEN_KV = 1024
SEQ_LEN_Q = 1
HEAD_DIM = 128
DEVICE = 'cuda'
TOLERANCE = 1e-4

# Kernel types for flash attention
KERNELS = {0:'fa_2'}

# Naive PyTorch implementation
def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y


def test_correctness(
    dtype: torch.dtype = torch.float32,
    rtol: float = TOLERANCE,
    atol: float = TOLERANCE,
    kernel_choices: Optional[Tuple[int, ...]] = None,
):
    """Test correctness of flash attention implementations against PyTorch."""
    if kernel_choices is None:
        kernel_choices = range(len(KERNELS))
        
    # Create random input
    q = torch.randn((BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM), dtype=dtype).cuda()
    k = torch.randn((BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM), dtype=dtype).cuda()
    v = torch.randn((BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM), dtype=dtype).cuda()
    
    # Torch output 
    torch_output = manual_attn(q, k, v)
    
    print("Checking correctness of Flash Attention")
    print(f"Batch size: {BATCH_SIZE}, Num heads: {NUM_HEADS}, Seq len Q: {SEQ_LEN_Q}, Seq len KV: {SEQ_LEN_KV}, Head dim: {HEAD_DIM}")
    
    # Test all kernel implementations
    for kernel_choice in kernel_choices:
        kernel_name = KERNELS[kernel_choice]
        dlcu_output = dlcu.flash_attention_forward(q, k, v, 1.0 / math.sqrt(HEAD_DIM), kernel_choice)
        
        # Verify correctness
        is_close = torch.allclose(torch_output, dlcu_output, rtol=rtol, atol=atol)
        max_diff = torch.max(torch.abs(torch_output - dlcu_output))
        print(f"Flash Attention kernel {kernel_name} correct: {is_close}, max diff: {max_diff:.6e}\n")
        assert is_close, f"Flash Attention kernel {kernel_name} failed correctness test with max diff {max_diff:.6e}"


def benchmark(
    dtype: torch.dtype = torch.float32,
    kernel_choices: Optional[Tuple[int, ...]] = None,
):
    """Benchmark flash attention implementations against PyTorch."""
    
    if kernel_choices is None:
        kernel_choices = range(len(KERNELS))
    
    # Warm up
    for kernel_choice in kernel_choices:
        for _ in range(10):
            q = torch.randn((BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM), dtype=dtype).cuda()
            k = torch.randn((BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM), dtype=dtype).cuda()
            v = torch.randn((BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM), dtype=dtype).cuda()
            _ = dlcu.flash_attention_forward(q, k, v, 1.0 / math.sqrt(HEAD_DIM), kernel_choice)
            _ = manual_attn(q, k, v)
    
    print('Profiling Flash Attention')
    print(f"Batch size: {BATCH_SIZE}, Num heads: {NUM_HEADS}, Seq len Q: {SEQ_LEN_Q}, Seq len KV: {SEQ_LEN_KV}, Head dim: {HEAD_DIM}")
    
    # Create random inputs
    q = torch.randn((BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM), dtype=dtype).cuda()
    k = torch.randn((BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM), dtype=dtype).cuda()
    v = torch.randn((BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM), dtype=dtype).cuda()
    
    print('=== Profiling Manual Attention ===')
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    manual_result = manual_attn(q, k, v)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Elapsed time: {elapsed_time_ms:.6f} ms")
    
    for kernel_choice in kernel_choices:
        kernel_name = KERNELS[kernel_choice]
        print(f'=== Profiling {kernel_name} ===')
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        dlcu_output = dlcu.flash_attention_forward(q, k, v, 1.0 / math.sqrt(HEAD_DIM), kernel_choice)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"Elapsed time: {elapsed_time_ms:.6f} ms")

        
if __name__ == "__main__":
    # Test correctness with default parameters
    test_correctness()
    
    # Benchmark with different sequence lengths
    benchmark()
