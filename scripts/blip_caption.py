import argparse
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--model", default="Salesforce/blip-image-captioning-base")
    args = p.parse_args()
    processor = BlipProcessor.from_pretrained(args.model)
    model = BlipForConditionalGeneration.from_pretrained(args.model)
    image = Image.open(args.image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=40)
    print(processor.decode(output[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
