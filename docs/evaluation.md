# Evaluating Vision-Language Models

[← Documentation home](README.md) · [Hands-on scoring](evaluation-hands-on.md)

## Goal

Separate four independent dimensions: **task quality**, **hallucinations**, **latency**, and **resource use**. A fluent answer is not necessarily correct, and a fast model is not necessarily useful.

## 1. Design a documented test set

| Subtask | Suggested baseline | Reference | Appropriate metric |
| --- | --- | --- | --- |
| Captioning | 20 licensed varied images | Human-reviewed factual attributes | Human grounding rubric; optional established caption metrics |
| Visual Q&A | 20 questions | Short verified answer | Exact/normalized match only for unambiguous answers |
| OCR | 20 clean and noisy samples | Character-accurate transcription | Character error rate (CER), word error rate (WER) |
| Charts | 10 charts with labeled values | Exact numeric answer | Numeric accuracy with explicit tolerance |
| Safety-critical observations | Diverse failure cases | Human verified | Recall/precision per task plus abstentions |

These sample counts are **small educational examples**, not enough to prove generalization or production accuracy.

## 2. Create an evaluation manifest

Use one JSONL record per question:

```json
{"id":"ocr-001","task":"ocr","image":"samples/receipt_001.jpg","prompt":"Transcribe visible text.","reference":"TOTAL 124.00","license":"owned","split":"test"}
{"id":"vqa-001","task":"vqa","image":"samples/table_001.jpg","prompt":"How many chairs are visible?","reference":"4","license":"owned","split":"test"}
```

The paths and answers above are **schema illustrations**, not bundled datasets or measured examples.

## 3. OCR scoring mathematics

For reference length `N`, character error rate is `CER = (substitutions + deletions + insertions) / N`; values can exceed 1 when insertions dominate. Word error rate uses the same edit-distance principle over whitespace-tokenized words. Define case, punctuation and whitespace normalization *before* testing.

```python
def edit_distance(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        curr = [i]
        for j, b in enumerate(hyp, 1):
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j-1] + (a != b)))
        prev = curr
    return prev[-1]

def cer(reference, predicted):
    if not reference:
        raise ValueError("CER undefined for empty reference")
    return edit_distance(reference, predicted) / len(reference)

# Deliberately synthetic test: one substitution among three reference characters.
assert cer("cat", "car") == 1 / 3
```

For a full command-line harness and validation of an evaluation file, see [Hands-on Evaluation](evaluation-hands-on.md).

## 4. Hallucination and qualitative rubric

Mark each answer as:
- **Grounded:** each material object/action/number is visible.
- **Incomplete:** relevant visible evidence omitted.
- **Unsupported:** asserts content not visible or not legible.
- **Ambiguous:** reference or image is insufficient to judge.

Record verbatim model text, image hash and reviewer identity. Distinguish *model uncertainty* from an *annotator disagreement*.

## 5. Performance methodology

- Report cold model load separately from warm end-to-end inference.
- Warm up the same checkpoint and shape; synchronize CUDA before/after timing.
- Report image dimensions, preprocessed visual token policy, input prompt, output token count, decoding configuration and precision.
- Compare same tasks under equivalent settings, and note models that cannot complete inference.
- Record temperatures and shared system memory for Jetson; PyTorch peak allocated CUDA memory is **not** total RAM.

## 6. Correct observations vs overclaims

| Observation | Supported conclusion | Unsupported conclusion |
| --- | --- | --- |
| BLIP generated a fluent caption | Decoding produced text | Caption is factually accurate |
| One Qwen run OOMed on 8GB | This exact configuration failed | Qwen can never run on Jetson |
| Reduced image resolution lowered latency | Under test conditions, less processing was needed | Resolution reduction never harms quality |
| One model scored higher on ten examples | Better score on these ten examples | Universally better model |
| Jetson reports 67 INT8 TOPS | Published compute capability | 67 tokens/sec |

## 7. Transparency

**Repository status:** the hardware results table is still empty pending physical measurements. Do not populate numerical performance claims without corresponding raw logs and reproducible commands.

See [Jetson Lab](../jetson/README.md), [Results template](../results/experiment-template.md), and [Optimization](optimization.md).
