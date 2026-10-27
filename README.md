# Getting Started with Vision-Language Models (VLMs)

A practical, open-source learning path from **BLIP image captioning** to **SmolVLM and Qwen visual reasoning**, with a dedicated **NVIDIA Jetson Orin Nano Super 8GB** edge-inference track.

> **Status:** Educational examples. Hardware performance is not yet independently measured. Do not interpret hardware TOPS as VLM token throughput.

## Learning tracks

| Track | Audience | Outcome |
|---|---|---|
| Beginner | Python learners | Image captioning and visual question answering |
| Intermediate | ML developers | Load, prompt, and compare open VLMs |
| Advanced | Edge AI engineers | Measure latency, memory, and throughput on Jetson |

## Models

| Model | Main task | Edge guidance |
|---|---|---|
| Salesforce BLIP base | Captioning / VQA | Start here |
| Hugging Face SmolVLM-256M-Instruct | General image reasoning | Small-memory candidate |
| Qwen2.5-VL-3B-Instruct | OCR, documents, visual reasoning | Requires memory profiling; quantization may help |
| Meta Llama 3.2 Vision 11B | General vision reasoning | Reference only: not a practical native 8GB baseline |

## Quick start

Python 3.10+ is recommended on a standard desktop or cloud GPU. For Jetson, use an NVIDIA-supported JetPack/PyTorch combination rather than blindly installing desktop CUDA wheels.

```bash
git clone https://github.com/Niraj-Lunavat/Getting-Started-with-VLM.git
cd Getting-Started-with-VLM
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/blip_caption.py --image path/to/image.jpg
```

## Repository map

- [Fundamentals](docs/architecture.md)
- [Six-week roadmap](docs/learning-roadmap.md)
- [Model comparison](docs/model-guide.md)
- [BLIP captioning](scripts/blip_caption.py)
- [SmolVLM example](scripts/smolvlm_infer.py)
- [Qwen example](scripts/qwen_infer.py)
- [Jetson guide](jetson/README.md)
- [Benchmarking](scripts/benchmark.py)
- [Contributing](CONTRIBUTING.md)

## Benchmark policy

Report model revision, precision, JetPack version, torch/transformers versions, power mode, input image resolution, prompt length, generated token count, peak allocated memory, process RSS, and latency. Warm up before timing. Record time-to-first-token separately when supported. **No fabricated FPS or tokens/s.**

## Safety and licensing

Check each model's upstream license and gated access requirements. Do not upload confidential medical, employee, or customer images to external services without authorization. Treat VLM answers as fallible, especially OCR, clinical interpretation, and safety-critical inspection.

## Upstream references

- [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base)
- [SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)
- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [Llama 3.2 Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)
- [Jetson Orin Nano Super](https://developer.nvidia.com/embedded/jetson-orin-nano-super-developer-kit)
