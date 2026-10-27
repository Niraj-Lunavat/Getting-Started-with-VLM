# Hands-on: Reproducible VLM Scoring

[← Documentation home](README.md) · [Evaluation methodology](evaluation.md)

This lab scores **saved predictions** against reference answers. It intentionally separates *answer quality* from *model inference performance*: no hardware or model run is claimed.

## Step 1 — Create a results CSV

```csv
id,task,reference,prediction
ocr-001,ocr,HELLO WORLD,HELLO W0RLD
vqa-001,vqa,4,4
vqa-002,vqa,red,blue
```

These rows are **synthetic demonstrations only**. In real experiments, keep the corresponding image hashes, prompts and model revisions in your dataset metadata.

## Step 2 — Run this Python evaluator

Save as `evaluate_predictions.py`, or copy the following into a Python shell:

```python
import csv
import sys
from collections import defaultdict

def levenshtein(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1,
                               previous[j-1]+(x != y)))
        previous = current
    return previous[-1]

def norm(s):
    return " ".join(s.casefold().split())

totals = defaultdict(lambda: {"distance": 0, "length": 0, "correct": 0, "count": 0})
with open(sys.argv[1], newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        task = row["task"].lower()
        ref, pred = norm(row["reference"]), norm(row["prediction"])
        record = totals[task]
        record["count"] += 1
        if task == "ocr":
            if not ref:
                raise ValueError("Empty OCR reference: " + row["id"])
            record["distance"] += levenshtein(ref, pred)
            record["length"] += len(ref)
        elif task == "vqa":
            record["correct"] += int(ref == pred)
        else:
            raise ValueError("Unknown task: " + task)
for task, item in sorted(totals.items()):
    if task == "ocr":
        print(f"OCR pooled CER: {item['distance']/item['length']:.4f}")
    else:
        print(f"VQA exact match: {item['correct']/item['count']:.4f}")
```

```bash
python evaluate_predictions.py saved_predictions.csv
```

## Expected output for the synthetic CSV

```text
OCR pooled CER: 0.0909
VQA exact match: 0.5000
```

These are **deterministic calculations on made-up reference/prediction strings**, not achieved BLIP, Qwen or SmolVLM model scores. The OCR reference contains 11 characters and the prediction one substitution.

## Step 3 — Interpret results

- **Pooled CER** = sum of edit distances / sum of reference characters. State normalization and language.
- **VQA exact match** only suits short answers with one canonical form; semantic variants can be correct yet fail this metric.
- A higher sample count is necessary for meaningful comparisons; report per-category breakdowns and confidence intervals for consequential claims.
- Review hallucinations manually against original images. Automated correctness cannot replace reference validation.

## Step 4 — Extend

Add model revision, JSONL image provenance, per-image latency, OCR WER, numerical tolerance for chart Q&A and a human evaluator rubric. Compare only matched test samples.

References: [Architecture](architecture.md) · [Model guide](model-guide.md) · [Jetson lab](../jetson/README.md).
