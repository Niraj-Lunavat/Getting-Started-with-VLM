"""SmolVLM image+text inference. Use with a compatible Transformers release."""
import argparse
from common import load_image, metadata, save_json

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", default="Describe this image accurately, without guessing.")
    p.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--output-json")
    args = p.parse_args()
    if args.max_new_tokens < 1:
        p.error("--max-new-tokens must be >= 1")
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    image = load_image(args.image)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForVision2Seq.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.prompt}]}]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt_text, images=[image], return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated = output[:, inputs["input_ids"].shape[1]:]
    answer = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    print(answer)
    if args.output_json:
        save_json(args.output_json, {**metadata(), "task": "image_qa", "model": args.model,
                                     "device": device, "image": args.image,
                                     "prompt": args.prompt, "answer": answer})

if __name__ == "__main__":
    main()
