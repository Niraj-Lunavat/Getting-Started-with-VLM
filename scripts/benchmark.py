"""Simple repeated-process latency benchmark for a VLM CLI.

Example:
  python scripts/benchmark.py --runs 5 -- python scripts/blip_caption.py --image sample.jpg

Includes process startup and model loading; not tokens/second.
"""
import argparse
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", default="results/latency.json")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command or args.runs < 1:
        parser.error("Provide a command after -- and --runs >= 1")
    times = []
    for index in range(args.runs):
        start = time.perf_counter()
        result = subprocess.run(command, capture_output=True, text=True)
        duration = time.perf_counter() - start
        if result.returncode:
            raise RuntimeError(f"Run {index+1} failed: {result.stderr[-1000:]}")
        times.append(duration)
        print(f"Run {index+1}: {duration:.3f}s")
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "measurement": "cold_process_end_to_end_seconds_including_model_loading",
        "runs_seconds": times,
        "mean_seconds": statistics.mean(times),
        "median_seconds": statistics.median(times),
        "warning": "Not a warm inference or token throughput benchmark."
    }
    from pathlib import Path
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"Saved {path}")

if __name__ == "__main__":
    main()
