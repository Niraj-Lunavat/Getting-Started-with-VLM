# 📚 VLM Documentation Hub

**Start here** to progress from image captioning to production-minded multimodal and edge-AI engineering.

This documentation is organized by *concept → runnable experiment → evaluation → hardware deployment*. Every performance claim must be tagged **measured**, **calculated**, **estimated**, or **not measured**.

## Choose your route

| Your goal | Read first | Then run |
| --- | --- | --- |
| Understand how a VLM works | [Architecture](architecture.md) | [BLIP captioning](../scripts/blip_caption.py) |
| Choose between BLIP, SmolVLM, Qwen, and Llama Vision | [Model guide](model-guide.md) | [SmolVLM](../scripts/smolvlm_infer.py) or [Qwen](../scripts/qwen_infer.py) |
| Follow a guided curriculum | [Learning roadmap](learning-roadmap.md) | Weekly labs |
| Compare accuracy and hallucinations | [Evaluation guide](evaluation.md) | [Evaluation harness](evaluation-hands-on.md) |
| Run on 8GB Jetson | [Jetson deployment guide](../jetson/README.md) | [Warm BLIP benchmark](../scripts/benchmark_blip_warm.py) |
| Understand deployment trade-offs | [Optimization and deployment](optimization.md) | Controlled tests |

## Recommended order

1. [Architecture: pixels to tokens and text](architecture.md)
2. [Models: capabilities, licensing and realistic memory needs](model-guide.md)
3. [Six-week beginner-to-engineer roadmap](learning-roadmap.md)
4. [Evaluation theory and methodology](evaluation.md)
5. [Hands-on evaluation: reproducible scoring code](evaluation-hands-on.md)
6. [Optimization: quantization, visual tokens, profiling](optimization.md)
7. [Jetson reproducibility protocol](../jetson/README.md)

## What is already measured?

**No physical Jetson model performance results have been submitted to this repository yet.** The existing scripts can generate real measurements on your hardware. Neither the Jetson's **up-to-67 INT8 TOPS** rating nor a model's parameter count determines its actual VLM tokens/second.

Use [the experiment template](../results/experiment-template.md) to submit raw logs, prompts, hardware setup, and observations.

## Repository code

- [BLIP caption](../scripts/blip_caption.py)
- [SmolVLM inference](../scripts/smolvlm_infer.py)
- [Qwen2.5-VL inference](../scripts/qwen_infer.py)
- [Cold-start benchmark](../scripts/benchmark.py)
- [Warm BLIP benchmark](../scripts/benchmark_blip_warm.py)

## Working with the docs

Each contribution should answer four questions: **What was tested? Under which configuration? What was observed? What remains uncertain?** Include runnable code and ground-truth data provenance; do not substitute illustrative numbers for measured performance.
