import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", default="Describe this image accurately.")
    args = p.parse_args()
    model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(args.image).convert("RGB")
    messages = [{"role":"user","content":[{"type":"image"},{"type":"text","text":args.prompt}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=100)
    print(processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0])

if __name__ == "__main__":
    main()
