import streamlit as st
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import io

model_path = "./model"

# Resize images to a consistent size
def preprocess_image(image, target_size=(512, 512)):
    image = image.resize(target_size)
    return np.array(image).astype(np.float32)[np.newaxis, ...] / 255.

def transfer_style(content_image, style_image, model_path):
    content_image = preprocess_image(content_image)
    style_image = preprocess_image(style_image, target_size=(256, 256))  # keep style 256x256 as expected by model
    hub_module = hub.load(model_path)
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0].numpy()
    stylized_image = (stylized_image * 255).astype(np.uint8)
    return stylized_image[0]

def main():
    st.set_page_config(page_title="Neural Style Transfer", layout="centered")
    st.markdown("<h1 style='text-align: center; color: #6c63ff;'>🎨 Artistic Style Transfer</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Upload a content image and a style image to generate art!</h4>", unsafe_allow_html=True)

    with st.container():
        st.markdown("### Upload Your Images:")
        col1, col2 = st.columns(2)

        with col1:
            content_file = st.file_uploader("📸 Content Image", type=["png", "jpg", "jpeg"], key="content")
        with col2:
            style_file = st.file_uploader("🖼️ Style Image", type=["png", "jpg", "jpeg"], key="style")

    if content_file and style_file:
        content_image = Image.open(content_file).convert("RGB")
        style_image = Image.open(style_file).convert("RGB")

        st.markdown("### 📷 Preview of Uploaded Images")
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(content_image, caption="Content Image", width=300)
        with img_col2:
            st.image(style_image, caption="Style Image", width=300)

        if st.button("✨ Transfer Style"):
            with st.spinner("Applying style..."):
                result = transfer_style(content_image, style_image, model_path)
                result_image = Image.fromarray(result)
                st.success("Done!")
                st.image(result_image, caption="🎉 Stylized Output", width=400)

if __name__ == "__main__":
    main()
