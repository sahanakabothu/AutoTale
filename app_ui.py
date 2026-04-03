from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# load model (first time download avtundi)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# load image
image_path = "images/img1.jpg"
image = Image.open(image_path).convert("RGB")
image = image.resize((512, 512))

# process image
inputs = processor(image, return_tensors="pt")

# generate caption
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)

print("🖼️ Caption:", caption)