import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

st.title("🖼️ AutoTale - Image Caption Generator")

# load model (only once)
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((512, 512))

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        st.success(f"Caption: {caption}")