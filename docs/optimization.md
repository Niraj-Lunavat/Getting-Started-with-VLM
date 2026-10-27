# VLM Optimization and Deployment Engineering

[← Documentation home](README.md) · [Jetson lab](../jetson/README.md)

## Optimization is a constrained experiment

Define your target (accuracy, latency, memory, power, sustained temperature) **before** changing precision or resolution. Optimizing one dimension frequently hurts another.

## 1. What consumes memory?

Approximate decomposition:

```text
Total memory ≈ model weights
             + activations / temporary workspaces
             + KV cache
             + image tensors and visual features
             + framework allocator overhead
             + other operating-system/process memory
```

For a parameter count `P` and `b` bits per weight, idealized raw weight bytes are `P × b / 8`. This is a **lower-bound calculation**, not observed resident RAM. Quantization metadata and unquantized layers add overhead. On Jetson, system and GPU workloads share the 8GB RAM pool.

## 2. Practical optimization knobs

| Change | Potential benefit | Potential cost / caveat |
| --- | --- | --- |
| FP16 or BF16 | Smaller weights vs FP32 | Hardware and kernel support must be checked |
| 8-bit / 4-bit weights | Lower weight memory | Compatibility, accuracy changes, dequantization overhead |
| Lower image resolution | Fewer pixels and often fewer visual tokens | Small objects/text may vanish |
| Shorter max output | Lower generation time and cache growth | Truncated or incomplete answers |
| Reuse loaded model | Remove model-load latency | Resident memory remains occupied |
| Faster attention/runtime kernels | Potential speed increase | Build/dependency and model support constraints |
| Limit concurrent requests | Protect shared memory | Reduced throughput under load |

**Caution:** CUDA memory allocators and multiple image crops/patches complicate memory estimation. Do not assume raw weight calculation predicts fit.

## 3. Code: calculate raw weight lower bound

```python
def decimal_gb_for_weights(parameters, bits_per_weight):
    if parameters <= 0 or bits_per_weight <= 0:
        raise ValueError("Inputs must be positive")
    return parameters * bits_per_weight / 8 / 1e9

for precision_bits in (32, 16, 8, 4):
    print(precision_bits, decimal_gb_for_weights(3_000_000_000, precision_bits))
```

Expected mathematical output: 12 GB, 6 GB, 3 GB and 1.5 GB respectively. **None of these are measured 8GB Jetson allocations.**

## 4. Benchmark protocol

1. Confirm JetPack, torch CUDA support, model revision and quantized runtime.
2. Record power mode and run `tegrastats` in another terminal.
3. Warm up the model at your target resolution and prompt shape.
4. Synchronize the GPU when timing and repeat ≥5 runs.
5. Log successful outputs, OOMs, timeouts, quality regressions and precision.
6. Repeat at a second image resolution to inspect sensitivity.
7. Compare quality on *identical evaluation inputs*.

## 5. Analysis checklist

A meaningful observation includes a number, units, method, environment and limitation. Example of acceptable **report structure**, not a result:

> "At [resolution], [model revision], [precision] and [JetPack version], median warm end-to-end latency across [N] runs was [measured seconds]. Peak system RAM was [measured MiB] according to tegrastats. [Limitations]."

Do not present your answer as a benchmark without running it. See the [results template](../results/experiment-template.md).

## 6. Edge design patterns

- **Periodic image analysis:** sample frames only when events occur rather than invoking a VLM on every video frame.
- **Hybrid CV + VLM:** fast object detector filters regions; VLM handles selective descriptions or operator Q&A.
- **OCR validation:** run structured verification on totals, dates or known vocabulary after visual text extraction.
- **Fallback service:** local lightweight model first, escalate selectively to a larger authorized server model if privacy allows.

These are architectural options; their quality and performance benefits need application-specific validation.
