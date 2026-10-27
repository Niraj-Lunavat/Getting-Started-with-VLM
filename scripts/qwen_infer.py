import argparse
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Local image path")
    p.add_argument("--prompt", default="Describe this image.")
    args = p.parse_args()
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(model_id)
    messages = [{"role":"user","content":[{"type":"image","image":"file://" + __import__("os").path.abspath(args.image)},{"type":"text","text":args.prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=128)
    ids = [o[len(i):] for i,o in zip(inputs.input_ids, ids)]
    print(processor.batch_decode(ids, skip_special_tokens=True)[0])

if __name__ == "__main__":
    main()
