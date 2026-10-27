# Jetson Orin Nano Super 8GB — reproducible VLM engineering lab

> **Evidence status (October 2026):** No benchmark in this repository has yet been executed on a physical Jetson. The tables below contain **test plans, not observed results**.

## 1. Hardware reality

The NVIDIA Jetson Orin Nano Super Developer Kit has **8GB unified CPU/GPU memory**, with NVIDIA advertising **up to 67 INT8 TOPS** in its Super configuration. TOPS is a theoretical hardware throughput metric and **cannot** be converted to VLM tokens/second or frames/second. Actual speed depends on visual tokens, resolution, model architecture, precision, kernels, prefill/decode split, thermals and memory pressure.

A 3B model at FP16 needs approximately **6 GB just for 3 billion parameters**, before the vision encoder, runtime, KV cache, activations and OS. This is a lower-bound estimate, **not measured allocation**. Quantization reduces weight memory but does not eliminate other overhead.

## 2. Recommended experiment order

| Priority | Model | Task | Expected engineering risk | Status |
|---|---|---|---|---|
| 1 | BLIP base | Image captioning | Lower | Not measured |
| 2 | SmolVLM 256M | Visual Q&A | Lower than multi-billion-parameter models; compatibility must be verified | Not measured |
| 3 | Qwen2.5-VL 3B | OCR / image reasoning | High memory pressure; test quantized runtimes | Not measured |
| Reference | Llama 3.2 Vision 11B | Multimodal reasoning | Impractical as straightforward local 8GB baseline | Not measured |

Do **not** interpret the ordering as a published benchmark ranking.

## 3. Record the exact platform

```bash
cat /etc/nv_tegra_release
uname -a
python3 --version
free -h
df -h
which tegrastats
sudo nvpmodel -q
python3 - <<'PY'
import torch, transformers
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime:", torch.version.cuda)
PY
```

Install an **NVIDIA-supported JetPack-compatible PyTorch build** first. A generic desktop pip wheel is not guaranteed to support CUDA on Jetson aarch64. Verify with `torch.cuda.is_available()` before running GPU experiments.

## 4. Collect resource telemetry

In a separate terminal:

```bash
mkdir -p results
tegrastats --interval 1000 --logfile results/jetson_tegrastats.log
```

Stop `tegrastats` after the run. The log records platform telemetry; it is **not automatically parsed into a validated peak-memory metric**. Do not equate PyTorch CUDA allocated memory with total shared system RAM.

## 5. Run a controlled first experiment

```bash
python scripts/blip_caption.py --image samples/your_image.jpg

# Existing benchmark: repeated *cold process* timings,
# including startup, model download/cache lookup and model loading.
python scripts/benchmark.py --runs 5 --output results/blip_cold.json -- \
  python scripts/blip_caption.py --image samples/your_image.jpg
```

Use a legally shareable image and record its resolution and SHA256. Downloading models during a timed run contaminates latency: pre-download first, or explicitly label the run as including download time.

## 6. What to measure

| Metric | Definition | Measurement caveat |
|---|---|---|
| Cold process latency | Spawn subprocess → command exits | Includes startup/model loading |
| Warm end-to-end latency | Input already available, model resident → output ready | Requires an in-process benchmark, not current subprocess tool |
| Time to first token | Generation request → first generated token | Requires streaming/instrumentation |
| Decode throughput | Generated tokens / decode duration | Must exclude prefill and count generated tokens |
| Peak PyTorch GPU allocated | `torch.cuda.max_memory_allocated()` | Excludes non-PyTorch and some shared allocations |
| Shared memory / thermals | Jetson `tegrastats` | Correlate telemetry timestamps with test window |
| Output quality | Exact/semantic comparison to annotated ground truth | Human evaluation recommended for open captions |

For fairness: use the same image, resolution policy, prompts, max output tokens, model revision, power mode and thermal stabilization. **Task-specific BLIP and chat-based VLMs are not directly comparable on a single generic quality score.**

## 7. Results ledger (unmeasured)

| Model | Precision | Cold median (s) | Warm p50 (s) | Decode tok/s | Peak system RAM | Evidence |
|---|---|---:|---:|---:|---:|---|
| BLIP base | TBD | — | — | N/A for captioning comparison until instrumented | — | Pending |
| SmolVLM 256M | TBD | — | — | — | — | Pending |
| Qwen2.5-VL 3B | TBD | — | — | — | — | Pending |

A dash means **not measured**, never zero. Do not add invented example numbers to this table.

## 8. Interpreting observations correctly

- **Out-of-memory** does not prove the model is unsupported; check background RAM use, precision, image tokens and KV cache. Record the failing configuration.
- **A model loads successfully** does not prove it can complete generation at a useful resolution or token count.
- **One fast run** does not establish stable throughput. Repeat warmed trials and report median, dispersion, and thermal conditions.
- **Low PyTorch allocated memory** does not mean the system has plenty of free RAM on a unified-memory Jetson.
- **Captioning quality** must be judged against image-grounded reference descriptions; fluent but incorrect outputs are failures.

## 9. Experiment report template

Copy [the results template](../results/experiment-template.md) for every run. Attach raw logs and state whether each figure is measured, calculated, or estimated.

## References

- [NVIDIA Orin Nano Super](https://developer.nvidia.com/embedded/jetson-orin-nano-super-developer-kit)
- [NVIDIA Jetson AI Lab](https://www.jetson-ai-lab.com/)
- [BLIP model card](https://huggingface.co/Salesforce/blip-image-captioning-base)
- [SmolVLM model card](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)
- [Qwen2.5-VL model card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
