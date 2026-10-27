"""BLIP visual question answering uses its own VQA checkpoint."""
import argparse
from common import load_image, metadata, save_json

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--model", default="Salesforce/blip-vqa-base")
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--output-json")
    args = p.parse_args()
    if args.max_new_tokens < 1:
        p.error("--max-new-tokens must be >= 1")
    import torch
    from transformers import BlipForQuestionAnswering, BlipProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = load_image(args.image)
    processor = BlipProcessor.from_pretrained(args.model)
    model = BlipForQuestionAnswering.from_pretrained(args.model).to(device).eval()
    inputs = processor(image, args.question, return_tensors="pt").to(device)
    with torch.inference_mode():
        tokens = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    answer = processor.decode(tokens[0], skip_special_tokens=True).strip()
    print(answer)
    if args.output_json:
        save_json(args.output_json, {**metadata(), "task": "vqa", "model": args.model,
                                     "device": device, "image": args.image,
                                     "question": args.question, "answer": answer})

if __name__ == "__main__":
    main()
