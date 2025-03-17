import torch
import time
import numpy as np
import dlcu
from typing import Tuple, Optional

# Macros to test for correctness
BATCH_SIZE = 8
SEQ_LEN = 128
HIDDEN_SIZE = 768
DEVICE = 'cuda'
TOLERANCE = 1e-4

# Kernel types for residual connection
KERNELS = {0: 'naive', 1: 'vectorised'}

def test_correctness(
    dtype: torch.dtype = torch.float32,
    rtol: float = TOLERANCE,
    atol: float = TOLERANCE,
    kernel_choices: Optional[Tuple[int, ...]] = None,
) -> None:
    """Test correctness of residual implementations against PyTorch."""
    if kernel_choices is None:
        kernel_choices = range(len(KERNELS))
    
    # Create random input
    x = torch.randn((BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE), dtype=dtype, device=DEVICE)
    residual = torch.randn((BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE), dtype=dtype, device=DEVICE)
    
    # PyTorch implementation is just addition
    torch_output = x + residual
    
    print("Checking correctness of Residual Connection")
    print(f"Batch size: {BATCH_SIZE}, Seq len: {SEQ_LEN}, Hidden size: {HIDDEN_SIZE}")
    
    # Test all kernel implementations
    for kernel_choice in kernel_choices:
        kernel_name = KERNELS[kernel_choice]
        dlcu_output = dlcu.residual_forward(x, residual, kernel_choice)
        
        # Verify correctness
        is_close = torch.allclose(torch_output, dlcu_output, rtol=rtol, atol=atol)
        max_diff = torch.max(torch.abs(torch_output - dlcu_output))
        print(f"Residual kernel {kernel_name} correct: {is_close}, max diff: {max_diff:.6e}")
        assert is_close, f"Residual kernel {kernel_name} failed correctness test with max diff {max_diff:.6e}"

def benchmark(
    dtype: torch.dtype = torch.float32,
    kernel_choices: Optional[Tuple[int, ...]] = None,
) -> None:
    """Benchmark residual implementations against PyTorch."""
    if kernel_choices is None:
        kernel_choices = range(len(KERNELS))
        
    # Warm up
    for kernel_choice in kernel_choices:
        for _ in range(10):
            x = torch.randn((BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE), dtype=dtype, device=DEVICE)
            residual = torch.randn((BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE), dtype=dtype, device=DEVICE)
            _ = dlcu.residual_forward(x, residual, kernel_choice)
            _ = x + residual
    
    x = torch.randn((BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE), dtype=dtype, device=DEVICE)
    residual = torch.randn((BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE), dtype=dtype, device=DEVICE)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    torch_output = x + residual
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
        dlcu_output =dlcu.residual_forward(x, residual, kernel_choice)
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
