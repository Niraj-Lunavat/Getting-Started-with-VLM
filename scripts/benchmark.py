"""Measure subprocess wall-clock latency, including interpreter and model loading."""
import argparse
import statistics
import subprocess
import time
from common import metadata, save_json

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--timeout", type=float, default=1200, help="Max seconds per run")
    p.add_argument("--output", default="results/cold_process.json")
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command or args.runs < 1 or args.timeout <= 0:
        p.error("Provide a command after --, --runs >= 1, and --timeout > 0")
    timings = []
    for index in range(args.runs):
        started = time.perf_counter()
        result = subprocess.run(command, text=True, capture_output=True, timeout=args.timeout)
        seconds = time.perf_counter() - started
        if result.returncode:
            raise RuntimeError(f"Run {index+1} failed (code {result.returncode}): {result.stderr[-1500:]}")
        timings.append(seconds)
        print(f"Run {index+1}/{args.runs}: {seconds:.3f}s; output={result.stdout.strip()[:200]!r}")
    report = {**metadata(), "measurement": "subprocess_wall_clock_including_model_loading",
              "command": command, "runs_seconds": timings,
              "median_seconds": statistics.median(timings),
              "mean_seconds": statistics.mean(timings),
              "min_seconds": min(timings), "max_seconds": max(timings),
              "caveat": "Repeated cold processes; weights may be cached. Not warm latency or decode tok/s."}
    save_json(args.output, report)
    print(f"Saved {args.output}")

if __name__ == "__main__":
    main()
