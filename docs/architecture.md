# VLM fundamentals

A vision-language model combines a visual encoder with a language model or text decoder. The visual pathway converts images into representations consumed by the language component. Architectures differ: BLIP uses task-specific vision-language components, while instruction-tuned VLMs accept chat-style image-and-text prompts.

## Core tasks
- Image captioning
- Visual question answering
- OCR and document understanding
- Chart reasoning
- Video frame understanding

## Failure modes
Hallucinated objects, small-text OCR errors, prompt sensitivity, visual grounding errors, and excessive confidence. Evaluate against labeled images before deploying.
