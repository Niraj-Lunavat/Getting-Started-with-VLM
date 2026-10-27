"""Warm BLIP inference benchmark with real measurements (no synthetic results).

Usage:
  python scripts/benchmark_blip_warm.py --image /path/to/image.jpg --warmup 2 --runs 10
Requires a working PyTorch + Transformers installation. On Jetson install JetPack-compatible torch.
"""
import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--model", default="Salesforce/blip-image-captioning-base")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--output", default="results/blip_warm.json")
    args = p.parse_args()
    if args.runs < 1 or args.warmup < 0:
        p.error("--runs must be positive and --warmup nonnegative")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(args.image).convert("RGB")
    processor = BlipProcessor.from_pretrained(args.model)
    model = BlipForConditionalGeneration.from_pretrained(args.model).to(device).eval()
    inputs = processor(images=image, return_tensors="pt").to(device)

    def generate():
        with torch.inference_mode():
            return model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    for _ in range(args.warmup):
        generate()
        synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    durations = []
    captions = []
    for _ in range(args.runs):
        synchronize(device)
        start = time.perf_counter()
        output = generate()
        synchronize(device)
        durations.append(time.perf_counter() - start)
        captions.append(processor.decode(output[0], skip_special_tokens=True))

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "measurement": "warm_in_process_full_generate_latency",
        "model": args.model,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "image_width_height": list(image.size),
        "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup,
        "runs_seconds": durations,
        "median_seconds": statistics.median(durations),
        "mean_seconds": statistics.mean(durations),
        "min_seconds": min(durations),
        "max_seconds": max(durations),
        "caption_each_run": captions,
        "peak_torch_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "limitations": "Includes image encoder + autoregressive generation; excludes model load and image preprocessing. Does not measure first token, decode-only tok/s, or total Jetson unified RAM.",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({"median_seconds": report["median_seconds"], "caption": captions[-1], "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
