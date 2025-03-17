import torch
import time
import dlcu
from typing import Tuple, Optional

# Macros to test for correctness
BATCH_SIZE = 8
SEQ_LEN = 128
IN_FEATURES = 1024
OUT_FEATURES = 768
DEVICE = 'cuda'
TOLERANCE = 1e-4

# Kernel types for linear layer
KERNELS = {0: 'naive', 1: 'shared_mem', 2: 'blocktiling_2d'}

def test_correctness(
    dtype: torch.dtype = torch.float32,
    rtol: float = TOLERANCE,
    atol: float = TOLERANCE,
    kernel_choices: Optional[Tuple[int, ...]] = None,
):
    """Test correctness of linear implementations against PyTorch."""
    if kernel_choices is None:
        kernel_choices = range(len(KERNELS))
    
    # Create random input
    x = torch.randn((BATCH_SIZE, SEQ_LEN, IN_FEATURES), dtype=dtype, device=DEVICE)
    weight = torch.randn((OUT_FEATURES, IN_FEATURES), dtype=dtype, device=DEVICE)
    bias = torch.randn(OUT_FEATURES, dtype=dtype, device=DEVICE)
    
    # PyTorch functional implementation
    torch_output = torch.nn.functional.linear(x, weight, bias)
    
    print("Checking correctness of Linear")
    print(f"Batch size: {BATCH_SIZE}, Seq len: {SEQ_LEN}, In features: {IN_FEATURES}, Out features: {OUT_FEATURES}")
    
    # Test all kernel implementations
    for kernel_choice in kernel_choices:
        kernel_name = KERNELS[kernel_choice]
        dlcu_output = dlcu.linear_forward(x, weight, bias, kernel_choice)
        
        # Verify correctness
        is_close = torch.allclose(torch_output, dlcu_output, rtol=rtol, atol=atol)
        max_diff = torch.max(torch.abs(torch_output - dlcu_output))
        print(f"Linear kernel {kernel_name} correct: {is_close}, max diff: {max_diff:.6e}")
        assert is_close, f"Linear kernel {kernel_name} failed correctness test with max diff {max_diff:.6e}"


def benchmark(
    dtype: torch.dtype = torch.float32,
    kernel_choices: Optional[Tuple[int, ...]] = None,
):
    """Benchmark linear implementations against PyTorch."""
    
    if kernel_choices is None:
        kernel_choices = range(len(KERNELS))
    
    # Warm up 
    for kernel_choice in kernel_choices:
        for _ in range(10):
            x = torch.randn((BATCH_SIZE, SEQ_LEN, IN_FEATURES), dtype=dtype, device=DEVICE)
            weight = torch.randn((OUT_FEATURES, IN_FEATURES), dtype=dtype, device=DEVICE)
            bias = torch.randn(OUT_FEATURES, dtype=dtype, device=DEVICE)
            _ = dlcu.linear_forward(x, weight, bias, kernel_choice)
            _ = torch.nn.functional.linear(x, weight, bias)
    
    print('Profiling Linear')
    print(f"Batch size: {BATCH_SIZE}, Seq len: {SEQ_LEN}, In features: {IN_FEATURES}, Out features: {OUT_FEATURES}")
    print('=== Profiling PyTorch Linear ===')
    
    x = torch.randn((BATCH_SIZE, SEQ_LEN, IN_FEATURES), dtype=dtype, device=DEVICE)
    weight = torch.randn((OUT_FEATURES, IN_FEATURES), dtype=dtype, device=DEVICE)
    bias = torch.randn(OUT_FEATURES, dtype=dtype, device=DEVICE)
    _ = dlcu.linear_forward(x, weight, bias, kernel_choice)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    torch_output = torch.nn.functional.linear(x, weight, bias)
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
        dlcu_output = dlcu.linear_forward(x, weight, bias, kernel_choice)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"Elapsed time: {elapsed_time_ms:.6f} ms")
        
        
if __name__ == "__main__":
    # Test correctness with default parameters
    test_correctness()
    
    # Benchmark with different configurations
    benchmark()
    
    # Example with different feature sizes (uncomment to use)
    # for in_out in [(768, 3072), (3072, 768), (1024, 4096), (4096, 1024)]:
    #     global IN_FEATURES, OUT_FEATURES
    #     IN_FEATURES, OUT_FEATURES = in_out
    #     benchmark()
