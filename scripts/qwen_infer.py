"""Qwen2.5-VL image inference; a 3B checkpoint may not fit an 8GB Jetson unquantized."""
import argparse
from common import image_path, metadata, save_json

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", default="Describe only what is visible in this image.")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--max-pixels", type=int, default=512 * 512, help="Processor pixel cap")
    p.add_argument("--output-json")
    args = p.parse_args()
    if args.max_new_tokens < 1 or args.max_pixels < 1:
        p.error("--max-new-tokens and --max-pixels must be >= 1")
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    path = image_path(args.image)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": path.as_uri()},
        {"type": "text", "text": args.prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    output = [tokens[len(input_ids):] for input_ids, tokens in zip(inputs.input_ids, output)]
    answer = processor.batch_decode(output, skip_special_tokens=True)[0].strip()
    print(answer)
    if args.output_json:
        save_json(args.output_json, {**metadata(), "task": "image_qa", "model": args.model,
                                     "image": str(path), "prompt": args.prompt,
                                     "answer": answer, "max_pixels": args.max_pixels,
                                     "device_map": str(model.hf_device_map) if hasattr(model, "hf_device_map") else None})

if __name__ == "__main__":
    main()
