# Evaluating VLMs responsibly

## Distinguish model tasks
BLIP captioning generates short descriptions. SmolVLM and Qwen instruction models answer free-form visual questions. A single latency/quality score across these tasks is misleading.

## Test suite proposal
1. **Captioning:** 20 licensed images with human reference captions. Evaluate object presence, relations, and hallucinations.
2. **VQA:** 20 questions with grounded short answers; report exact-match accuracy only where exact match is meaningful.
3. **OCR:** 20 documents with reference transcription; report character error rate and word error rate.
4. **Charts:** 10 plots with manually verified numeric questions; score correctness with tolerances.
5. **Edge stress:** repeated generation at multiple resolutions, varying output lengths, and steady thermal state.

Record provenance, consent, image size, prompt, raw output, expected output, and reviewer notes. Do not use sensitive patient or employee images in public benchmarks.

## Results taxonomy
- **Measured:** obtained from executed code and attached raw logs.
- **Calculated:** mathematically derived from explicitly identified inputs (e.g. parameter count × bytes/parameter).
- **Estimated:** projection based on assumptions, not device observation.
- **Not measured:** no experimental evidence available.

## Common misleading conclusions
- Faster first run can reflect cached weights or a shorter response.
- Tokens/second can increase when image resolution or prompt length changes, without model improvement.
- Accuracy on a handful of examples is not statistically robust.
- A valid-looking answer may contain hallucinated details.

See [Jetson experiment protocol](../jetson/README.md) and [results template](../results/experiment-template.md).
