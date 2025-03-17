# PyTorch Tests

## Running Tests For Kernels

Just run the python file for the desired operation in the `deep-learning.cu/csrc/kernels` directory. For example running tests for LayerNorm:

```
python test_layernorm.py
```

An example output (NVIDIA RTX A4000):
```
Checking correctness of LayerNorm
N: 1024, Hidden size: 768
LayerNorm kernel naive correct: True, max diff: 5.340576e-05
LayerNorm kernel shared_mem correct: True, max diff: 4.959106e-05
LayerNorm kernel intra_warp correct: True, max diff: 4.863739e-05
LayerNorm kernel variance_estimate correct: True, max diff: 4.863739e-05
Profiling LayerNorm
N: 1024, Hidden size: 768
=== Profiling PyTorch LayerNorm ===
Elapsed time: 0.017408 ms
=== Profiling naive ===
Elapsed time: 0.690176 ms
=== Profiling shared_mem ===
Elapsed time: 0.041088 ms
=== Profiling intra_warp ===
Elapsed time: 0.020480 ms
=== Profiling variance_estimate ===
Elapsed time: 0.020480 ms
```


