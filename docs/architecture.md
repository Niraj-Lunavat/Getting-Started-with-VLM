# Architecture of Vision-Language Models

[← Documentation home](README.md) · [Next: model guide →](model-guide.md)

## 1. What makes a VLM different?

Text-only language models map text tokens to text predictions. A VLM adds image processing so it can condition text generation on visual content. Some VLMs specialize in *captioning or VQA*, while newer *instruction-tuned multimodal language models* follow conversational prompts and can reason over images.

```text
Photo (RGB pixels)                  Text instruction
        │                                  │
 Image preprocessing                Text tokenizer
        │                                  │
    Vision encoder                         │
        │                                  │
 Visual embeddings                         │
        └──────────┬───────────────────────┘
                   ▼
          Fusion / projection
                   ▼
          Text decoder or LLM
                   ▼
          Generated answer
```

This is a **conceptual** diagram, not the exact internal architecture of all BLIP, SmolVLM, Qwen or Llama variants.

## 2. From pixels to visual tokens

Typical processing includes resizing/cropping, normalization, patch embedding, transformer-based visual encoding and a projector/resampler into the language model's feature space. Image resolution is a major memory and latency variable. Some models employ dynamic image tiling or adaptive visual token allocation; avoid assuming one fixed number of image tokens.

A 448×448 image with non-overlapping 14×14 patches yields `(448/14)^2 = 1024` **patches** in this simplified example. This is **not** a claim that any of our selected models produces exactly 1,024 final visual tokens: pooling, tiling, special tokens and compression change counts.

## 3. Three common architectures

| Pattern | General idea | Why it matters |
| --- | --- | --- |
| Dual encoders | Separate image and text representations aligned by similarity learning | Retrieval and representation learning |
| Encoder-decoder vision-language models | Image features condition a text decoder | BLIP captioning-style tasks |
| Vision encoder + projector + autoregressive LLM | Image features enter an instruction-tuned generative language model | Conversational VQA, OCR, chart Q&A |

**BLIP** combines vision-language pretraining with task-specific components. **SmolVLM**, **Qwen-VL** and **Llama Vision** are multimodal families with distinct encoders, projectors, tokenization conventions and model-specific chat templates. Study the precise architecture of the exact checkpoint before drawing comparisons.

## 4. The forward pass: what runs when?

1. **Preprocess:** decode image, resize/normalize, create image tensors.
2. **Encode:** transform image tensors into visual features.
3. **Fuse/prefill:** combine visual context with prompt tokens and initialize generation state.
4. **Decode:** predict successive output tokens, usually autoregressively.
5. **Postprocess:** decode token IDs into human-readable text.

Time-to-first-token includes input processing and prefill, whereas **decode tokens/s** measures subsequent token generation. A one-number total latency measurement cannot isolate these bottlenecks.

## 5. Code: inspect your runtime

```python
import torch
import transformers

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

For hands-on generation:
```bash
python scripts/blip_caption.py --image /path/to/image.jpg
python scripts/smolvlm_infer.py --image /path/to/image.jpg --prompt "List visible objects."
python scripts/qwen_infer.py --image /path/to/image.jpg --prompt "Read the visible text."
```
Run commands from the repository root. Install a compatible environment first; Jetson requires JetPack-compatible PyTorch builds.

## 6. Failure analysis

- **Hallucination:** model invents non-visible objects; verify against image evidence.
- **OCR error:** wrong character sequences in small, rotated or blurred text.
- **Spatial confusion:** left/right, counts and relationships may be mistaken.
- **Prompt sensitivity:** slightly different phrasing changes answer.
- **Resolution dependence:** more pixels can improve small-text recognition but may add visual tokens, memory and latency.

## 7. Exercise

Choose five images you can legally share (indoor, outdoor, text-heavy, multi-object, difficult lighting). Generate a caption per image. Record visible objects, omissions and hallucinations. Your expected result is **a reproducible qualitative log**, not a guaranteed accuracy score.

**Further reading:** [BLIP paper](https://arxiv.org/abs/2201.12086), [Qwen2.5-VL model card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), [SmolVLM model card](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct).
