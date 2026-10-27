# Six-Week Learning Roadmap — Beginner to Edge AI Engineer

[← Documentation home](README.md)

**Format:** 6 modules, roughly 4–8 hours/week depending on prerequisites. Every week produces a small commit and an evidence-backed report. This is a curriculum, not a performance guarantee.

| Week | Core objective | Practical deliverable | Completion criteria |
| --- | --- | --- | --- |
| 1 | VLM architecture, image preprocessing, visual tokens | Architecture notes and dataset manifest | Explain prefill vs decode, resolution trade-offs |
| 2 | BLIP image captioning and VQA | Caption and VQA examples | 10 labeled images; error log |
| 3 | SmolVLM prompting | Prompt comparison experiment | 3 prompt variants, 10 images, raw outputs |
| 4 | Qwen2.5-VL document understanding | OCR/VQA comparison | Reference transcription and documented errors |
| 5 | Performance fundamentals and quantization | Jetson environment and model-fit analysis | Reproducible environment record |
| 6 | Benchmark and publish | Results table, raw logs, README contribution | Measured vs calculated claims labeled correctly |

## Week 1 — Fundamentals

Read [Architecture](architecture.md). Record image dimensions, file formats and how preprocessing can change visible detail.

```python
from pathlib import Path
from PIL import Image

for p in Path("samples").glob("*"):
    try:
        with Image.open(p) as im:
            print(p.name, im.size, im.mode)
    except OSError:
        pass
```

**Observation to document:** a photo's pixel resolution is not equivalent to its final model visual token count.

## Week 2 — BLIP

```bash
python scripts/blip_caption.py --image samples/example.jpg
```

Write ten ground-truth caption notes. Categorize hallucinations, misses, wrong colors and object counts. For direct questions use the BLIP VQA code in [Model Guide](model-guide.md).

## Week 3 — SmolVLM

```bash
python scripts/smolvlm_infer.py --image samples/example.jpg --prompt "Describe only what is visible."
python scripts/smolvlm_infer.py --image samples/example.jpg --prompt "Count the visible objects. Say if unsure."
```

Record outputs verbatim; do not treat a single prompt's stronger response as a statistically meaningful win.

## Week 4 — Qwen and OCR

```bash
python scripts/qwen_infer.py --image samples/document.jpg --prompt "Transcribe the text exactly; mark unreadable segments."
```

Ground truth should be typed from the source image and separately reviewed. Compute text errors using [Evaluation Hands-on](evaluation-hands-on.md).

## Week 5 — Edge AI

Read [Optimization](optimization.md) and the [Jetson guide](../jetson/README.md). Record JetPack, PyTorch CUDA support, memory and power mode. Do not attempt all models at once on an 8GB device.

## Week 6 — Reproducible results

```bash
python scripts/benchmark_blip_warm.py --image samples/example.jpg --warmup 2 --runs 10
```

Attach your actual JSON and [experiment report](../results/experiment-template.md). A measured latency figure must state whether model loading, preprocessing and generation are included.

## Capstone choices

- **Factory PPE audit assistant:** explain potential PPE observations, human verify; not a certified safety system.
- **Invoice/document extraction:** OCR plus validation against known fields.
- **Accessibility captioning:** descriptive assistance with disclosure of limitations.
- **Edge VLM selection:** compare latency, memory and answer quality using consistent inputs.

Avoid using identifiable patient, employee or confidential client imagery in public repositories.
