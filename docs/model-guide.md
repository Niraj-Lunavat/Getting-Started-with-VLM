# Choosing a VLM: BLIP, SmolVLM, Qwen and Llama Vision

[← Documentation home](README.md) · [Architecture](architecture.md)

## Comparison matrix

| Family/checkpoint | Primary learning task | Example code | Jetson Orin Nano Super 8GB notes |
| --- | --- | --- | --- |
| [BLIP captioning base](https://huggingface.co/Salesforce/blip-image-captioning-base) | Generate an image caption | [Script](../scripts/blip_caption.py) | Best first experiment in this repo; device results pending |
| [BLIP VQA base](https://huggingface.co/Salesforce/blip-vqa-base) | Ask a direct visual question | See code below | Separate checkpoint from captioning |
| [SmolVLM 256M Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) | Instruction-following on images | [Script](../scripts/smolvlm_infer.py) | Small model; Jetson compatibility/performance must be validated |
| [Qwen2.5-VL 3B Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | OCR, document understanding, visual Q&A | [Script](../scripts/qwen_infer.py) | High 8GB memory pressure; quantized runtime may be required |
| [Llama 3.2 Vision 11B Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) | Larger multimodal assistant | Reference model | Not a practical native 8GB starting point; upstream access/license applies |

**Model variants matter.** Do not generalize performance from one version of Qwen-VL, SmolVLM or BLIP to the whole family. Check model-card supported transformer versions and hardware requirements. Use only checkpoints whose licenses suit your application.

## Minimal BLIP caption example

```python
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

model_id = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_id)
model = BlipForConditionalGeneration.from_pretrained(model_id)
image = Image.open("sample.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")
tokens = model.generate(**inputs, max_new_tokens=40)
print(processor.decode(tokens[0], skip_special_tokens=True))
```

**Result:** The program prints one model-generated caption. Its exact wording depends on the input photo, weights, library versions and decoding settings; **no output caption is asserted here without a real test image**.

## BLIP visual question answering

Use the **VQA checkpoint**, not the captioning checkpoint:

```python
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

model_id = "Salesforce/blip-vqa-base"
processor = BlipProcessor.from_pretrained(model_id)
model = BlipForQuestionAnswering.from_pretrained(model_id)
image = Image.open("sample.jpg").convert("RGB")
question = "How many people are visible?"
inputs = processor(image, question, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=20)
print(processor.decode(output[0], skip_special_tokens=True))
```

Evaluate counts manually against ground truth; models can miscount occluded or small objects.

## SmolVLM and Qwen: code and cautions

```bash
python scripts/smolvlm_infer.py --image sample.jpg --prompt "Describe the image and state uncertainty."
python scripts/qwen_infer.py --image sample.jpg --prompt "Transcribe all readable text without guessing."
```

The repository scripts demonstrate APIs, but successful inference depends on suitable processor/model versions, access to weights, memory availability and device compatibility. Qwen's `device_map="auto"` may offload parts of the model, which changes the benchmark conditions.

## Memory lower bounds (calculated, not measured)

For a checkpoint with **N billion parameters**, raw parameter storage is roughly:
- FP32: `4 × N` GB
- FP16/BF16: `2 × N` GB
- INT8: `1 × N` GB
- 4-bit packed: `0.5 × N` GB

For 3B parameters, FP16 weights alone are about **6 GB decimal**. This ignores quantization scales, vision components if not counted in the quoted parameter figure, KV cache, activations, runtime workspace and shared-memory system overhead. 4-bit deployment requires actual compatible kernels and should never be assumed to work just because theoretical weights fit.

## Decision guide

**Start BLIP** for deterministic task exploration and dataset validation. **Try SmolVLM** when memory is constrained and instruction following is needed. **Test Qwen** when OCR and richer visual prompting matter, after memory profiling. **Study Llama Vision** on a suitable larger GPU or hosted environment if licensing/access permits.

## What counts as results?

| Claim | Evidence required |
| --- | --- |
| "Runs on Jetson" | JetPack, memory mode, exact model revision, successful raw command output |
| "Model A is faster" | Matched input, warm timing, repetitions, precision, power mode |
| "Model A is more accurate" | Same labeled dataset, scoring rubric and failures |
| "4-bit reduces RAM" | Quantized checkpoint, actual device memory measurements |

No benchmark figures have been verified for this repository's physical Jetson yet. See [Evaluation](evaluation.md) and [Jetson Lab](../jetson/README.md).
