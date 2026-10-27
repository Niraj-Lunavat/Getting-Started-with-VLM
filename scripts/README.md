# VLM inference and benchmarking scripts

Runnable command-line examples for **BLIP**, **BLIP VQA**, **SmolVLM**, and **Qwen2.5-VL**. Run commands from the repository **root directory**.

> **Evidence status:** Source syntax and lightweight helper tests were validated locally. Real inference outputs and Jetson hardware performance are **not yet verified**. No model performance metrics are invented.

## Script directory

| File | Purpose | Output |
| --- | --- | --- |
| [blip_caption.py](blip_caption.py) | Caption an image with BLIP base | Text, optional JSON |
| [blip_vqa.py](blip_vqa.py) | Answer a visual question with BLIP VQA checkpoint | Text, optional JSON |
| [smolvlm_infer.py](smolvlm_infer.py) | Instruct a lightweight VLM | Text, optional JSON |
| [qwen_infer.py](qwen_infer.py) | Visual reasoning/OCR with Qwen2.5-VL | Text, optional JSON |
| [benchmark.py](benchmark.py) | Repeated subprocess wall-clock timing | JSON |
| [benchmark_blip_warm.py](benchmark_blip_warm.py) | Warm model-resident BLIP timing | JSON |
| [common.py](common.py) | Shared image validation and JSON utilities | Library |

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

On **Jetson Orin Nano Super 8GB**, install an NVIDIA-supported **JetPack-compatible PyTorch** build first. Generic PyPI wheels may not expose CUDA on aarch64.

## Quick start

Use an image file you own or are authorized to process; change the path below:

```bash
python scripts/blip_caption.py --image samples/example.jpg --output-json results/blip_caption.json
python scripts/blip_vqa.py --image samples/example.jpg --question "How many people are visible?"
python scripts/smolvlm_infer.py --image samples/example.jpg --prompt "Describe the visible objects without guessing."
python scripts/qwen_infer.py --image samples/example.jpg --max-pixels 262144 --prompt "Transcribe the text without guessing."
```

Model weights download on first use. Qwen2.5-VL-3B may **not** fit unquantized on 8GB Jetson. Its `device_map="auto"` may offload layers; record actual placement before claiming GPU performance.

## Benchmarking: do not mix measurement types

```bash
# Includes process startup, checkpoint loading, inference and exit:
python scripts/benchmark.py --runs 5 --timeout 1200 --output results/blip_cold_process.json -- \
  python scripts/blip_caption.py --image samples/example.jpg

# Model already loaded; preprocessed input; synchronized generation:
python scripts/benchmark_blip_warm.py --image samples/example.jpg --warmup 2 --runs 10 \
  --output results/blip_warm.json
```

The cold **process** benchmark runs a fresh Python interpreter each time, but cached model files remain cached. It is **not** equivalent to a completely cold machine boot. The warm benchmark times the BLIP `generate()` call and omits loading/preprocessing. Neither script measures decode-only tokens/s or first-token latency. PyTorch CUDA allocations do not represent all unified Jetson RAM.

## Observations to record

| Observed behavior | Supported interpretation |
| --- | --- |
| A caption is fluent | Generation succeeded; accuracy still requires visual verification |
| A request fails with OOM | That exact device/configuration failed; not proof all quantized configurations fail |
| Qwen uses CPU offload | Latency is not all-GPU latency |
| First run is slow | Cache/download, model loading, and startup may dominate |
| Small image produces wrong OCR | Resolution may be insufficient; test accuracy/latency trade-off |

Attach raw output, image metadata, exact checkpoint, library versions, memory and power mode. Use [the experiment template](../results/experiment-template.md) for publication.

## Validate locally

```bash
python -m compileall -q scripts tests
python -m unittest discover -s tests -v
```

Tests do not download model weights. They validate only shared utilities; full inference requires separate environment/hardware tests.

For theory see [docs](../docs/README.md). For edge test procedures see [Jetson](../jetson/README.md).
