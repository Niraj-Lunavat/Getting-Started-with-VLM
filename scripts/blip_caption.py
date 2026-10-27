"""BLIP captioning, with CPU/CUDA selection and machine-readable output."""
import argparse
from common import load_image, metadata, save_json

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--output-json", help="Optional output record")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; check your PyTorch/JetPack build")
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto":
        device = "cpu"
    image = load_image(args.image)
    processor = BlipProcessor.from_pretrained(args.model)
    model = BlipForConditionalGeneration.from_pretrained(args.model).to(device).eval()
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    caption = processor.decode(ids[0], skip_special_tokens=True).strip()
    print(caption)
    if args.output_json:
        save_json(args.output_json, {**metadata(), "task": "caption", "model": args.model,
                                    "device": device, "image": args.image,
                                    "image_dimensions": list(image.size), "caption": caption})

if __name__ == "__main__":
    main()
