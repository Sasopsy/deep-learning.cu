import torch
import time
import numpy as np
import dlcu
from typing import Tuple, Optional
import time

# Macros to test for correctness
N = 1024
HIDDEN_SIZE = 768
DEVICE = 'cuda'
TOLERANCE = 1e-4

# Kernel types for layernorm
KERNELS = {0: 'naive', 1: 'shared_mem', 2: 'intra_warp', 3: 'variance_estimate'}

def test_correctness(
    dtype: torch.dtype = torch.float32,
    rtol: float = TOLERANCE,
    atol: float = TOLERANCE,
    kernel_choices: Optional[Tuple[int, ...]] = None,
) -> None:
    """Test correctness of layernorm implementations against PyTorch."""
    if kernel_choices is None:
        kernel_choices = range(len(KERNELS))
    
    # Create random input
    x = torch.randn((N, HIDDEN_SIZE), dtype=dtype, device=DEVICE)
    weight = torch.randn(HIDDEN_SIZE, dtype=dtype, device=DEVICE)
    bias = torch.randn(HIDDEN_SIZE, dtype=dtype, device=DEVICE)
    
    # PyTorch functional implementation
    torch_output = torch.nn.functional.layer_norm(x, [HIDDEN_SIZE], weight=weight, bias=bias)
    
    print("Checking correctness of LayerNorm")
    print(f"N: {N}, Hidden size: {HIDDEN_SIZE}")
    
    # Test all kernel implementations
    for kernel_choice in kernel_choices:
        kernel_name = KERNELS[kernel_choice]
        dlcu_output = dlcu.layernorm_forward(x, weight, bias, kernel_choice)
        
        # Verify correctness
        is_close = torch.allclose(torch_output, dlcu_output, rtol=rtol, atol=atol)
        max_diff = torch.max(torch.abs(torch_output - dlcu_output))
        print(f"LayerNorm kernel {kernel_name} correct: {is_close}, max diff: {max_diff:.6e}")
        assert is_close, f"LayerNorm kernel {kernel_name} failed correctness test with max diff {max_diff:.6e}"

def benchmark(
    dtype: torch.dtype = torch.float32,
    kernel_choices: Optional[Tuple[int, ...]] = None,
) -> None:
    """Benchmark layernorm implementations against PyTorch."""
    if kernel_choices is None:
        kernel_choices = range(len(KERNELS))
    
    # Warm up
    for kernel_choice in kernel_choices:
        for _ in range(10):
            x = torch.randn((N, HIDDEN_SIZE), dtype=dtype, device=DEVICE)
            weight = torch.randn(HIDDEN_SIZE, dtype=dtype, device=DEVICE)
            bias = torch.randn(HIDDEN_SIZE, dtype=dtype, device=DEVICE)
            _ = dlcu.layernorm_forward(x, weight, bias, kernel_choice)
            _ = torch.nn.functional.layer_norm(x, [HIDDEN_SIZE], weight=weight, bias=bias)
    
    print('Profiling LayerNorm')
    print(f"N: {N}, Hidden size: {HIDDEN_SIZE}")
    print('=== Profiling PyTorch LayerNorm ===')
    
    x = torch.randn((N, HIDDEN_SIZE), dtype=dtype, device=DEVICE)
    weight = torch.randn(HIDDEN_SIZE, dtype=dtype, device=DEVICE)
    bias = torch.randn(HIDDEN_SIZE, dtype=dtype, device=DEVICE)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    torch_output = torch.nn.functional.layer_norm(x, [HIDDEN_SIZE], weight=weight, bias=bias)
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
        dlcu_output = dlcu.layernorm_forward(x, weight, bias, kernel_choice)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"Elapsed time: {elapsed_time_ms:.6f} ms")

if __name__ == "__main__":
    # Test correctness with default parameters
    test_correctness()
    
    # Benchmark implementations
    benchmark()
    
    # Example with different hidden sizes (uncomment to use)
    # for hidden in [768, 1024, 2048, 4096]:
    #     global HIDDEN_SIZE
    #     HIDDEN_SIZE = hidden
    #     benchmark()
