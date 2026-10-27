# VLM experiment report — TEMPLATE (not measured)

## Experiment identity
- Date/time UTC:
- Engineer:
- Git commit:
- Hardware:
- JetPack / L4T:
- Power mode / clocks:
- Ambient temperature:
- PyTorch / Transformers / CUDA:
- Model ID and exact revision:
- Runtime / precision / quantization:
- Image SHA256 / resolution:
- Prompt / output token limit:
- Warm-up count / measured runs:

## Results — enter actual values only

| Metric | Value | Units | How measured |
|---|---|---|---|
| Cold process median | Not measured | seconds | subprocess benchmark |
| Warm end-to-end median | Not measured | seconds | in-process timer with CUDA synchronization |
| Time to first token | Not measured | seconds | streamer / instrumented generation |
| Decode throughput | Not measured | tokens/s | generated token count / decode seconds |
| Peak system RAM | Not measured | MiB | tegrastats |
| Peak PyTorch CUDA allocation | Not measured | MiB | torch.cuda.max_memory_allocated |
| Temperature | Not measured | °C | tegrastats |

## Qualitative evaluation

**Image and expected answer:** 

**Actual output (verbatim):**

**Grounded observations:** 

**Hallucinations or missed objects:**

**Failure cases / reproduction commands:**

**Conclusions supported by evidence:**

**Limitations and next experiment:**

> Never fill missing results with estimated values. Label analytical estimates explicitly.
