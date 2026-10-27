# NVIDIA Jetson Orin Nano Super 8GB

The kit offers **8GB shared CPU/GPU memory**. Its advertised **up to 67 INT8 TOPS** is not a measured VLM generation speed.

## Before installing
1. Check `cat /etc/nv_tegra_release`, `python3 --version`, `free -h`, `nvidia-smi` (if supported) and `tegrastats`.
2. Install the PyTorch build supported by your exact JetPack release; generic pip torch wheels may not enable CUDA on aarch64.
3. Start with BLIP or SmolVLM. Keep image sizes and output token counts modest.
4. Monitor memory with `tegrastats`; CPU and GPU share memory. A model fitting by parameter count alone does not guarantee successful inference.
5. Qwen2.5-VL-3B may require quantization, reduced image resolution and careful runtime selection; test compatibility first.
6. Do not assume Llama Vision 11B fits locally.

## Performance measurement
Record JetPack, power mode, clocks, temperature, model ID/revision, quantization, image resolution, prompt and generated tokens, end-to-end latency, warm latency, and peak RAM. Run at least 5 warmed trials. Avoid reporting FPS for autoregressive generation without defining it.

## References
[NVIDIA developer kit](https://developer.nvidia.com/embedded/jetson-orin-nano-super-developer-kit) · [Jetson AI Lab](https://www.jetson-ai-lab.com/)
